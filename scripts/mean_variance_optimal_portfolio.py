#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 22:32:03 2025

@author: dominikjurek
"""



# --- Standard Library Imports ---
import os
import warnings

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd
import yfinance as yf
import cvxopt as opt

import warnings
from arch.univariate.base import DataScaleWarning, ConvergenceWarning 
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, ctx


# --- Global Settings & Constants ---
# Set the working directory.
# NOTE: You may need to change this path to your project's root directory.
DIRECTORY = "."
os.chdir(DIRECTORY)

# Set the seed
np.random.seed(42)

# Analysis Period
ANALYSIS_YEARS = 15

# Optimization & Simulation Parameters
FRONTIER_POINTS = 50  # Number of points to calculate on the efficient frontier.
MC_SIM_SCENARIOS = 10000  # Number of scenarios for Monte Carlo simulation.
MC_SIM_HORIZON_MONTHS = 120  # 10-year horizon for simulation.
RESAMPLE_ITERATIONS = 100  # Number of bootstrap iterations for resampled frontier.
ROLLING_WINDOW_MONTHS = 60  # 5-year rolling window for dynamic weight analysis.

# Regime Modeling Parameters
MIN_OBS_PER_REGIME = 6  # Minimum data points required to consider a regime valid.
MAX_REGIMES_TO_TEST = 4  # Test models with 2 up to this number of regimes.

# --- Initial Setup ---
# Configure pandas for better display
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Configure the CVXOPT solver to not display progress messages
opt.solvers.options["show_progress"] = False

# ─── Warning filters ───
warnings.filterwarnings("ignore", category=UserWarning)            # generic statsmodels
warnings.filterwarnings("ignore", category=ConvergenceWarning)     # sklearn / statsmodels
warnings.filterwarnings("ignore", category=DataScaleWarning)       # ARCH data-scale
warnings.filterwarnings("ignore", category=RuntimeWarning)         # ARCH convergence


# Load immediately from local directory
df_etf_metadata = pd.read_csv("data/df_etf_metadata.csv")
etf_name_map = dict(zip(df_etf_metadata["Symbol"], df_etf_metadata["Fund name"]))
etf_expense_map = dict(zip(df_etf_metadata["Symbol"], df_etf_metadata["Expense ratio"]))
etf_symbols = list(etf_name_map.keys())

# To create a diversified portfolio of broad asset classes, we remove
# specialized, sector-specific ETFs and redundant funds.
industry_keywords = [
    "Energy", "Health Care", "Consumer", "Materials", "Financials",
    "Utilities", "Real Estate", "Industrials", "Communication", "Information Technology",
]
# List of specific ETFs to remove (often sector-focused or overlapping)
remove_symbols = [
    "VGT", "VHT", "VPU", "VDC", "VAW", "VIS", "VFH", "VNQ", "VOX", "VDE", "VCR",
]


def is_industry_or_redundant(symbol, name_map):
    """Checks if an ETF is sector-specific or on the removal list."""
    name = name_map.get(symbol, "")
    is_industry = any(keyword in name for keyword in industry_keywords)
    is_redundant = symbol in remove_symbols
    return is_industry or is_redundant


etf_symbols = [s for s in etf_symbols if not is_industry_or_redundant(s, etf_name_map)]
etf_symbols = list(dict.fromkeys(etf_symbols))  # Ensure unique symbols
print(f"\nFiltered down to {len(etf_symbols)} ETFs for analysis.")


# Load directly from local data
returns_monthly = pd.read_csv("data/returns_monthly.csv", index_col=0, parse_dates=True)



# Data Cleaning:
# Drop ETFs that do not have at least 10 years of non-NA observations
MIN_OBSERVATIONS = 10 * 12
returns_monthly = returns_monthly.dropna(axis=1, thresh=MIN_OBSERVATIONS)

# Drop any month (row) that still has missing values
returns_monthly = returns_monthly.dropna(axis=0)

# Update final list of ETFs and related data based on the cleaned DataFrame
etf_symbols = returns_monthly.columns.tolist()

# The S&P 500 ETF (VOO) is our primary benchmark; it's required for the analysis.
if "VOO" not in etf_symbols:
    raise ValueError(
        "VOO data is missing or was dropped. It is required for benchmark comparison."
    )

# Create a NumPy array of expense ratios in the same order as our final ETF symbols
expense_vector = np.array([etf_expense_map.get(sym, 0.0) for sym in etf_symbols])

print(
    f"\nFinal analysis will use {len(etf_symbols)} ETFs over {len(returns_monthly)} months."
)
print(
    f"Analysis period: {returns_monthly.index.min().date()} to {returns_monthly.index.max().date()}"
)


# Calculate historical annualized mean returns, net of expense ratios
annual_mu_sample = (returns_monthly.mean().values * 12) - expense_vector

# The sample covariance matrix is calculated from historical returns and annualized
sample_cov = returns_monthly.cov().values
annual_cov_sample = sample_cov * 12

voo_returns_monthly = returns_monthly["VOO"]
voo_mu_annual = voo_returns_monthly.mean() * 12 - etf_expense_map.get("VOO", 0.0)
voo_sigma_annual = voo_returns_monthly.std() * np.sqrt(12)


def select_portfolio(frontier, target_metric, target_value):
    """
    Selects a portfolio from the efficient frontier closest to a target value.

    Args:
        frontier (dict): The efficient frontier dictionary.
        target_metric (str): The metric to match ('mu' or 'sigma').
        target_value (float): The target return or volatility.

    Returns:
        tuple: Index and weights of the selected portfolio, or (None, None).
    """
    if not frontier[target_metric]:
        return None, None
    diffs = np.abs(np.array(frontier[target_metric]) - target_value)
    idx = diffs.argmin()
    return idx, frontier["weights"][idx]



def efficient_frontier(
    covariance_matrix, expected_returns, n_points=50, lambda_l1=0.0
):
    """
    Calculates the efficient frontier using the Markowitz model with optional L1
    regularization (LASSO). This encourages sparse portfolios by driving the
    weights of less important assets to exactly zero.

    This function reformulates the L1-regularized problem into a standard
    Quadratic Program (QP) that can be solved efficiently by CVXOPT.

    Args:
        covariance_matrix (np.array): Annualized covariance matrix of asset returns.
        expected_returns (np.array): Annualized vector of expected asset returns.
        n_points (int): The number of points to calculate along the frontier.
        lambda_l1 (float): The regularization strength. Higher values lead to
                           more sparsity (more zero weights).

    Returns:
        dict: A dictionary containing returns ('mu'), volatilities ('sigma'),
              and portfolio weights ('weights') for each point on the frontier.
    """
    n_assets = len(expected_returns)

    # We solve for a combined vector x = [w, u] of size 2*n_assets, where:
    # w: the standard portfolio weights (n_assets)
    # u: auxiliary variables to handle the absolute value |w_i| (n_assets)
    # The objective becomes: minimize 0.5*w'.Cov.w + lambda*1'.u
    # Subject to: w-u <= 0, -w-u <= 0 (which implies u >= |w|)

    # 1. The Quadratic Term P
    # Only involves 'w', so P_new has the original covariance_matrix in the
    # top-left block and zeros elsewhere.
    P_l1 = opt.matrix(
        np.block(
            [
                [covariance_matrix, np.zeros((n_assets, n_assets))],
                [np.zeros((n_assets, n_assets)), np.zeros((n_assets, n_assets))],
            ]
        )
    )

    # 2. The Linear Term q
    # The L1 penalty `lambda * sum(|w_i|)` is reformulated as `lambda * sum(u_i)`.
    # This becomes the linear part of the objective, q'.x.
    q_l1 = opt.matrix(np.concatenate([np.zeros(n_assets), lambda_l1 * np.ones(n_assets)]))

    # 3. The Inequality Constraints G and h (for Gx <= h)
    # These enforce u_i >= |w_i| and the original box constraints 0 <= w_i <= 1.
    I = np.eye(n_assets)
    Z = np.zeros((n_assets, n_assets))
    G_l1 = opt.matrix(
        np.block(
            [
                [I, -I],  # For w_i - u_i <= 0
                [-I, -I],  # For -w_i - u_i <= 0
                [-I, Z],  # For -w_i <= 0 (w_i >= 0)
                [I, Z],  # For w_i <= 1
            ]
        )
    )
    h_l1 = opt.matrix(np.concatenate([np.zeros(3 * n_assets), np.ones(n_assets)]))

    # 4. The Equality Constraints A and b (for Ax = b)
    # These constraints (sum of weights = 1, portfolio return = target)
    # only apply to the 'w' part of our variable vector 'x'.
    A_l1 = opt.matrix(
        np.block(
            [
                [expected_returns, np.zeros(n_assets)],
                [np.ones(n_assets), np.zeros(n_assets)],
            ]
        )
    )

    # Iterate through a range of target returns to trace the frontier
    target_mus = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    frontier = {"mu": [], "sigma": [], "weights": []}
    for target_mu in target_mus:
        # The RHS of the equality constraint
        b_l1 = opt.matrix([target_mu, 1.0])
        try:
            # Solve the larger, reformulated QP problem
            solution = opt.solvers.qp(P_l1, q_l1, G_l1, h_l1, A_l1, b_l1)
            if solution["status"] == "optimal":
                # Extract only the weights 'w' from the solution vector 'x'.
                weights = np.array(solution["x"][:n_assets]).flatten()
                # Clean up tiny weights due to solver precision
                weights[np.abs(weights) < 1e-7] = 0
                # Re-normalize to ensure sum is exactly 1 after cleanup
                if np.sum(weights) > 0:
                    weights /= np.sum(weights)

                sigma = np.sqrt(weights.T @ covariance_matrix @ weights)
                actual_mu = weights.T @ expected_returns
                frontier["mu"].append(actual_mu)
                frontier["sigma"].append(sigma)
                frontier["weights"].append(weights)
        except ValueError:
            # Solver may fail if no feasible solution exists for a target return
            pass
    return frontier


def prune_frontier(frontier):
    """
    Removes dominated portfolios from a mean-variance frontier, keeping only
    the efficient upper arm where return increases with volatility.

    Args:
        frontier (dict): A dict with keys 'sigma', 'mu', 'weights'.

    Returns:
        dict: A pruned frontier with the same structure.
    """
    vol = np.asarray(frontier["sigma"])
    ret = np.asarray(frontier["mu"])
    wlist = list(frontier["weights"])

    # 1) Sort by volatility (ascending)
    order = np.argsort(vol)
    vol, ret = vol[order], ret[order]
    wlist = [wlist[i] for i in order]

    # 2) Walk from left to right, keeping only points with strictly increasing returns
    keep_idx = []
    last_best_ret = -np.inf
    for i in range(len(ret)):
        if ret[i] > last_best_ret + 1e-12:
            keep_idx.append(i)
            last_best_ret = ret[i]

    # 3) Assemble the pruned frontier
    return {
        "sigma": vol[keep_idx].tolist(),
        "mu": ret[keep_idx].tolist(),
        "weights": [wlist[i] for i in keep_idx],
    }

# Frontier using simple sample estimates
ef_raw = prune_frontier(
    efficient_frontier(annual_cov_sample, annual_mu_sample, n_points=FRONTIER_POINTS)
)


# Find portfolios on each frontier matching the VOO benchmark's risk or return
_, w_mu_raw = select_portfolio(ef_raw, "mu", voo_mu_annual)
_, w_sigma_raw = select_portfolio(ef_raw, "sigma", voo_sigma_annual)


# --- Frontier arrays – ensure they exist ---

sigmas = np.asarray(ef_raw["sigma"])
mus = np.asarray(ef_raw["mu"])
weights = np.asarray(ef_raw["weights"])

symbols = globals().get("symbols", etf_symbols)
name_map = globals().get("name_map", etf_name_map)

# ── 0.  CONSTANTS ─── #
RESOLUTION = 0.0001         # one-basis-point step (≈0.01 %) for smooth sliders


# ──- 1.  BUILD STATIC PART OF THE FIGURE ─--- #

def build_initial_fig():
    fig = go.Figure()
    fig.update_layout(
        title="Efficient Frontier (Raw Estimates)",
        xaxis_title="Volatility (σ)",
        yaxis_title="Expected Return (µ)",
        hovermode="closest",
        clickmode="event+select",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
    )

    # frontier line
    fig.add_scatter(
        x=sigmas,
        y=mus,
        mode="lines",
        line=dict(color="lightgray"),
        hoverinfo="skip",
        showlegend=False,
    )

    # frontier points (trace 1)
    fig.add_scatter(
        x=sigmas,
        y=mus,
        mode="markers",
        marker=dict(size=7, color="gray", opacity=0.5),
        name="Raw Frontier",
        hoverlabel=dict(bgcolor="white"),
        hovertemplate="σ: %{x:.2%}<br>µ: %{y:.2%}<br>(click to see top weights)<extra></extra>",
    )

    # VOO reference (trace 2)
    fig.add_scatter(
        x=[voo_sigma_annual],
        y=[voo_mu_annual],
        mode="markers",
        marker=dict(
            symbol="diamond", size=14, color="royalblue", line=dict(width=2, color="black")
        ),
        name="VOO (ref)",
        hoverlabel=dict(bgcolor="white"),
        hovertemplate="VOO<br>σ: %{x:.2%}<br>µ: %{y:.2%}<extra></extra>",
    )

    # selected point (trace 3 — updated dynamically)
    mid = len(sigmas) // 2
    fig.add_scatter(
        x=[sigmas[mid]],
        y=[mus[mid]],
        mode="markers",
        marker=dict(size=14, color="red", line=dict(width=2, color="black")),
        name="Selected",
        hoverinfo="skip",
    )

    return fig

# ──- 2.  SMALL UTILS ─-- #

def top3_component(idx: int):
    w = weights[idx]
    top_idx = np.argsort(w)[-3:][::-1]
    items = [
        html.Li(
            [
                html.B(symbols[i]),
                html.Span(f" ({name_map.get(symbols[i],'Unknown')})",
                          style={"color": "#666"}),
                html.Span(f"{w[i]:.2%}", style={"float": "right"}),
            ]
        )
        for i in top_idx if w[i] > 0.001
    ]
    return html.Div(
        [
            html.H4(f"Top 3 ETFs – Frontier Point {idx+1}",
                    style={"margin": "4px 0 8px 0", "fontSize": "15px"}),
            html.Ul(items, style={"listStyle": "none", "paddingLeft": "0", "margin": "0"}),
        ],
        style={"fontFamily": "Arial, sans-serif", "fontSize": "14px",
               "lineHeight": "1.4", "maxWidth": "420px"},
    )


def closest_point(target_mu, target_sigma):
    """Return index of frontier point closest to (target_sigma,target_mu)."""
    dist = (mus - target_mu) ** 2 + (sigmas - target_sigma) ** 2
    return int(np.argmin(dist))


# ── 3.  LAYOUT ── #
app = dash.Dash(__name__)
fig_initial = build_initial_fig()
mid_idx = len(sigmas) // 2

app.layout = html.Div(
    [
        dcc.Graph(id="frontier-graph", figure=fig_initial, style={"height": "500px"}),
        html.Div(
            [
                html.Label("Target µ"),
                dcc.Slider(
                    id="mu-slider",
                    min=float(mus.min()),
                    max=float(mus.max()),
                    value=float(round(mus[mid_idx], 2)),
                    step=RESOLUTION,    # ← continuous & clickable
                    marks=None,           # ← hide grey numbers
                    updatemode="drag",
                    tooltip={"placement": "bottom"},
                ),
            ],
            style={"margin": "20px 0"},
        ),
        html.Div(
            [
                html.Label("Target σ"),
                dcc.Slider(
                    id="sigma-slider",
                    min=float(sigmas.min()),
                    max=float(sigmas.max()),
                    value=float(round(sigmas[mid_idx], 2)),
                    step=RESOLUTION,
                    marks=None,
                    updatemode="drag",
                    tooltip={"placement": "bottom"},
                ),
            ],
            style={"margin": "20px 0"},
        ),
        html.Div(
            [
                html.Button("Snap to VOO", id="snap-btn", n_clicks=0, className="btn"),
                dcc.Dropdown(
                    id="snap-criterion",
                    options=[
                        {"label": "Match µ (return)", "value": "mu"},
                        {"label": "Match σ (volatility)", "value": "sigma"},
                    ],
                    value="mu",
                    style={"width": "180px", "display": "inline-block", "marginLeft": "10px"},
                ),
            ],
            style={"margin": "10px 0"},
        ),
        html.Div(id="top-weights-box"),
    ],
    style={"width": "700px", "margin": "auto"},
)

# ── 4.  CALLBACKS ───────────────────────────────────────────────────────────── #
@app.callback(
    Output("frontier-graph", "figure"),
    Output("mu-slider", "value"),
    Output("sigma-slider", "value"),
    Output("top-weights-box", "children"),
    Input("frontier-graph", "clickData"),
    Input("mu-slider", "value"),
    Input("sigma-slider", "value"),
    Input("snap-btn", "n_clicks"),
    State("snap-criterion", "value"),
    prevent_initial_call=True,
)
def update_dash(click_data, mu_val, sigma_val, n_clicks, snap_choice):
    trigger = ctx.triggered_id
    sel_idx = mid_idx

    if trigger == "frontier-graph" and click_data:
        sel_idx = click_data["points"][0]["pointIndex"]
    elif trigger == "snap-btn" and n_clicks:
        sel_idx = int(np.argmin(np.abs(mus - voo_mu_annual))) if snap_choice == "mu" \
                  else int(np.argmin(np.abs(sigmas - voo_sigma_annual)))
    else:
        sel_idx = closest_point(mu_val, sigma_val)

    # update selected marker
    fig = build_initial_fig()
    fig.data[3].x = [sigmas[sel_idx]]
    fig.data[3].y = [mus[sel_idx]]

    # round slider feedback to two decimals
    mu_val_new    = round(float(mus[sel_idx]), 2)
    sigma_val_new = round(float(sigmas[sel_idx]), 2)

    html_output = top3_component(sel_idx)

    return fig, mu_val_new, sigma_val_new, html_output

# ── 5.  RUN ─────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    app.run_server(debug=True, port=8050, use_reloader=False)
