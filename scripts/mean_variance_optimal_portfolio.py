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
import cvxpy as cp

from pathlib import Path
from functools import lru_cache
from collections import deque

import plotly.graph_objects as go

import webbrowser
import threading
import dash
from dash import dcc, html, Input, Output, State, ctx
from flask import request
import threading, time, os

_last_ping = time.time()


# --- Global Settings & Constants ---
# Set the working directory.

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


# Configure pandas for better display
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)



CANDIDATE_SCRIPT_DIRS = ("scripts", "script")
REQUIRED_DATA_FILES = ("df_etf_metadata.csv", "returns_monthly.csv")

def looks_like_root(p: Path) -> bool:
    # Must have scripts/ (or script/) and data/
    if not any((p / d).is_dir() for d in CANDIDATE_SCRIPT_DIRS):
        return False
    data = p / "data"
    if not data.is_dir():
        return False
    # Require at least one expected file in data/ to avoid false positives
    return any((data / f).exists() for f in REQUIRED_DATA_FILES)

@lru_cache
def find_project_root() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()

    # 1) Walk upward
    for p in [start] + list(start.parents):
        if looks_like_root(p):
            return p
        # 2) Check each ancestor's immediate children (handles being in Investment/)
        for child in p.iterdir():
            if child.is_dir() and looks_like_root(child):
                return child

    # 3) Bounded downward BFS from CWD (depth ≤ 3)
    max_depth = 3
    q = deque([(Path.cwd().resolve(), 0)])
    visited = set()
    while q:
        node, depth = q.popleft()
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        if looks_like_root(node):
            return node
        if depth < max_depth:
            for ch in node.iterdir():
                if ch.is_dir():
                    q.append((ch, depth + 1))

    # 4) Optional env override (always wins)
    env = os.getenv("ETF_OPTIMIZER_ROOT")
    if env:
        q = Path(env).resolve()
        if looks_like_root(q):
            return q

    raise FileNotFoundError(f"Could not locate project root from start={start}")

def data_path(name: str) -> Path:
    return find_project_root() / "data" / name

# Usage
df_etf_metadata = pd.read_csv(data_path("df_etf_metadata.csv"))
returns_monthly = pd.read_csv(data_path("returns_monthly.csv"), index_col=0, parse_dates=True)


# Load immediately from local directory
df_etf_metadata = pd.read_csv(data_path("df_etf_metadata.csv"))
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
returns_monthly = pd.read_csv(data_path("returns_monthly.csv"), index_col=0, parse_dates=True)



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



def efficient_frontier(covariance_matrix, expected_returns, n_points=50, lambda_l1=0.0):
    """
    Calculates the efficient frontier using the Markowitz model with optional L1
    regularization (LASSO) via CVXPY. Long-only with 0 <= w <= 1 and sum(w)=1.

    Args:
        covariance_matrix (np.array or pd.DataFrame): Annualized covariance matrix (n x n).
        expected_returns (np.array or pd.Series): Annualized expected returns (n,).
        n_points (int): Number of target-return points to trace along the frontier.
        lambda_l1 (float): L1 regularization strength (higher => sparser weights).

    Returns:
        dict with:
          - 'mu': list of realized returns at each solution
          - 'sigma': list of volatilities at each solution
          - 'weights': list of np.array weights for each solution
    """
    # --- inputs -> numpy
    Sigma_in = np.asarray(covariance_matrix, dtype=float)
    mu = np.asarray(expected_returns, dtype=float).ravel()
    n = mu.shape[0]

    # Symmetrize & tiny ridge for numerical stability (solver only)
    Sigma = 0.5 * (Sigma_in + Sigma_in.T)
    Sigma = Sigma + 1e-10 * np.eye(n)

    # targets to trace (you can choose a different grid if you prefer)
    target_mus = np.linspace(mu.min(), mu.max(), n_points)

    frontier = {"mu": [], "sigma": [], "weights": []}

    for target in target_mus:
        w = cp.Variable(n)

        # Base constraints: long-only, fully invested
        cons_eq = [cp.sum(w) == 1, w >= 0, w <= 1, mu @ w == target]
        cons_ge = [cp.sum(w) == 1, w >= 0, w <= 1, mu @ w >= target]

        obj = cp.Minimize(cp.quad_form(w, Sigma) + lambda_l1 * cp.norm1(w))
        prob = cp.Problem(obj, cons_eq)

        # Try OSQP first (fast for QP), then SCS. If equality infeasible, relax to ≥.
        solved = False
        for constraints in (cons_eq, cons_ge):
            prob = cp.Problem(obj, constraints)
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
                if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
                    raise RuntimeError
                solved = True
            except Exception:
                try:
                    prob.solve(solver=cp.SCS, verbose=False)
                    solved = w.value is not None and prob.status in ("optimal", "optimal_inaccurate")
                except Exception:
                    solved = False
            if solved:
                break

        if not solved or w.value is None:
            # Skip this target if no feasible solution
            continue

        # Clean & renormalize
        w_sol = np.asarray(w.value).ravel()
        w_sol[np.abs(w_sol) < 1e-10] = 0.0
        s = w_sol.sum()
        w_sol = (w_sol / s) if s > 0 else np.ones(n) / n

        # Report with original Sigma (no ridge) for sigma
        sigma = float(np.sqrt(w_sol @ Sigma_in @ w_sol))
        mu_realized = float(w_sol @ mu)

        frontier["mu"].append(mu_realized)
        frontier["sigma"].append(sigma)
        frontier["weights"].append(w_sol)

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

@app.server.route("/shutdown", methods=["POST"])
def _shutdown():
    # Try dev-server shutdown; else hard-exit
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
    else:
        os._exit(0)
    return "Shutting down"

@app.server.route("/ping", methods=["POST"])
def _ping():
    global _last_ping
    _last_ping = time.time()
    return "ok"

def _watchdog(timeout=20):
    global _last_ping
    while True:
        time.sleep(5)
        if time.time() - _last_ping > timeout:
            os._exit(0)

threading.Thread(target=_watchdog, args=(20,), daemon=True).start()

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
        
        html.Script("""
            (function(){
            const post = (u,b) => {
                try {
                if (navigator.sendBeacon) return navigator.sendBeacon(u, new Blob([b||'.'], {type:'text/plain'}));
                fetch(u, {method:'POST', keepalive:true, body:b||'.'});
                } catch(e){}
            };
            const base = window.location.origin;
            // heartbeat every 5s
            const t = setInterval(()=>post(base + '/ping'), 5000);
            // try to shut down immediately when the tab/window goes away
            const bye = () => { try{ post(base + '/shutdown','bye'); }catch(e){} };
            window.addEventListener('pagehide', bye);
            window.addEventListener('beforeunload', bye);
            })();
            """)
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
    threading.Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:8050/")).start()
    app.run(debug=True, port=8050, use_reloader=False)