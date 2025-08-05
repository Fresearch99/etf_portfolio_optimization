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
from ipywidgets import FloatSlider, VBox, HTML, Button, Dropdown, HBox
from IPython.display import display


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



etf_name_map = {'VBIL': '0-3 Month Treasury Bill ETF',
 'VTEC': 'California Tax-Exempt Bond ETF',
 'VOX': 'Communication Services ETF',
 'VCR': 'Consumer Discretionary ETF',
 'VDC': 'Consumer Staples ETF',
 'VCRB': 'Core Bond ETF',
 'VCRM': 'Core Tax-Exempt Bond ETF',
 'VPLS': 'Core-Plus Bond ETF',
 'VIG': 'Dividend Appreciation ETF',
 'VWOB': 'Emerging Markets Government Bond ETF',
 'VDE': 'Energy ETF',
 'VSGX': 'ESG International Stock ETF',
 'VCEB': 'ESG U.S. Corporate Bond ETF',
 'ESGV': 'ESG U.S. Stock ETF',
 'EDV': 'Extended Duration Treasury ETF',
 'VXF': 'Extended Market ETF',
 'VFH': 'Financials ETF',
 'VEU': 'FTSE All-World ex-US ETF',
 'VSS': 'FTSE All-World ex-US Small-Cap ETF',
 'VEA': 'FTSE Developed Markets ETF',
 'VWO': 'FTSE Emerging Markets ETF',
 'VGK': 'FTSE Europe ETF',
 'VPL': 'FTSE Pacific ETF',
 'VNQI': 'Global ex-U.S. Real Estate ETF',
 'VGVT': 'Government Securities Active ETF',
 'VUG': 'Growth ETF',
 'VHT': 'Health Care ETF',
 'VYM': 'High Dividend Yield ETF',
 'VIS': 'Industrials ETF',
 'VGT': 'Information Technology ETF',
 'BIV': 'Intermediate-Term Bond ETF',
 'VCIT': 'Intermediate-Term Corporate Bond ETF',
 'VTEI': 'Intermediate-Term Tax-Exempt Bond ETF',
 'VGIT': 'Intermediate-Term Treasury ETF',
 'VIGI': 'International Dividend Appreciation ETF',
 'VYMI': 'International High Dividend Yield ETF',
 'VV': 'Large-Cap ETF',
 'BLV': 'Long-Term Bond ETF',
 'VCLT': 'Long-Term Corporate Bond ETF',
 'VTEL': 'Long-Term Tax-Exempt Bond ETF',
 'VGLT': 'Long-Term Treasury ETF',
 'VAW': 'Materials ETF',
 'MGC': 'Mega Cap ETF',
 'MGK': 'Mega Cap Growth ETF',
 'MGV': 'Mega Cap Value ETF',
 'VO': 'Mid-Cap ETF',
 'VOT': 'Mid-Cap Growth ETF',
 'VOE': 'Mid-Cap Value ETF',
 'VMBS': 'Mortgage-Backed Securities ETF',
 'VGMS': 'Multi-Sector Income Bond ETF',
 'MUNY': 'New York Tax-Exempt Bond ETF',
 'VNQ': 'Real Estate ETF',
 'VONE': 'Russell 1000 ETF',
 'VONG': 'Russell 1000 Growth ETF',
 'VONV': 'Russell 1000 Value ETF',
 'VTWO': 'Russell 2000 ETF',
 'VTWG': 'Russell 2000 Growth ETF',
 'VTWV': 'Russell 2000 Value ETF',
 'VTHR': 'Russell 3000 ETF',
 'VOO': 'S&P 500 ETF',
 'VOOG': 'S&P 500 Growth ETF',
 'VOOV': 'S&P 500 Value ETF',
 'IVOO': 'S&P Mid-Cap 400 ETF',
 'IVOG': 'S&P Mid-Cap 400 Growth ETF',
 'IVOV': 'S&P Mid-Cap 400 Value ETF',
 'VIOO': 'S&P Small-Cap 600 ETF',
 'VIOG': 'S&P Small-Cap 600 Growth ETF',
 'VIOV': 'S&P Small-Cap 600 Value ETF',
 'VSDB': 'Short Duration Bond ETF',
 'VSDM': 'Short Duration Tax-Exempt Bond ETF',
 'BSV': 'Short-Term Bond ETF',
 'VCSH': 'Short-Term Corporate Bond ETF',
 'VTIP': 'Short-Term Inflation-Protected Securities ETF',
 'VTES': 'Short-Term Tax-Exempt Bond ETF',
 'VGSH': 'Short-Term Treasury ETF',
 'VB': 'Small-Cap ETF',
 'VBK': 'Small-Cap Growth ETF',
 'VBR': 'Small-Cap Value ETF',
 'VTEB': 'Tax-Exempt Bond ETF',
 'BND': 'Total Bond Market ETF',
 'VTC': 'Total Corporate Bond ETF',
 'VTP': 'Total Inflation-Protected Securities ETF',
 'BNDX': 'Total International Bond ETF',
 'VXUS': 'Total International Stock ETF',
 'VTI': 'Total Stock Market ETF',
 'VTG': 'Total Treasury ETF',
 'BNDW': 'Total World Bond ETF',
 'VT': 'Total World Stock ETF',
 'VFMV': 'U.S. Minimum Volatility ETF',
 'VFMO': 'U.S. Momentum Factor ETF',
 'VFMF': 'U.S. Multifactor ETF',
 'VFQY': 'U.S. Quality Factor ETF',
 'VFVA': 'U.S. Value Factor ETF',
 'VUSB': 'Ultra-Short Bond ETF',
 'VGUS': 'Ultra-Short Treasury ETF',
 'VPU': 'Utilities ETF',
 'VTV': 'Value ETF'}

etf_symbols = list(etf_name_map.keys())

def get_total_return_series(ticker):
    """
    Fetches maximum available historical prices for a ticker from Yahoo Finance,
    adjusted for dividends and splits to represent total return.

    Args:
        ticker (str): The stock or ETF symbol.

    Returns:
        pd.DataFrame: A DataFrame of historical adjusted closing prices.
                      Returns an empty DataFrame on failure.
    """
    print(f"Downloading data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        # 'back_adjust=True' provides a total return series by adjusting historical
        # prices for both dividends and stock splits. 'auto_adjust=False' is required.
        df = stock.history(period="max", auto_adjust=False, back_adjust=True)[
            ["Close"]
        ].rename(columns={"Close": ticker})
        return df
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        return pd.DataFrame()


# Combine all price series into one DataFrame
all_prices_list = [get_total_return_series(ticker) for ticker in etf_symbols]
all_prices = pd.concat([df for df in all_prices_list if not df.empty], axis=1)

# Standardize the index to datetime objects without timezone information
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)

# Resample daily prices to month-end, then calculate monthly percentage returns
returns_monthly = all_prices.resample("ME").last().pct_change()

# Drop the last row if it's from the current (incomplete) month
last_date = returns_monthly.index[-1]
today = pd.Timestamp.today()

# Check if last observation is in the current month and year
if last_date.month == today.month and last_date.year == today.year:
    returns_monthly = returns_monthly.iloc[:-1]

# Limit data to the last N years for a more relevant analysis window
cutoff_date = returns_monthly.index.max() - pd.DateOffset(years=ANALYSIS_YEARS)
returns_monthly = returns_monthly[returns_monthly.index > cutoff_date]


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


print(
    f"\nFinal analysis will use {len(etf_symbols)} ETFs over {len(returns_monthly)} months."
)
print(
    f"Analysis period: {returns_monthly.index.min().date()} to {returns_monthly.index.max().date()}"
)

# Calculate historical annualized mean returns
annual_mu_sample = (returns_monthly.mean().values * 12)

# The sample covariance matrix is calculated from historical returns and annualized
sample_cov = returns_monthly.cov().values
annual_cov_sample = sample_cov * 12

ovariance_matrix = returns_monthly.cov() * 12

voo_returns_monthly = returns_monthly["VOO"]
voo_mu_annual = voo_returns_monthly.mean() * 12
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

# --- Build the figure ---

fig = go.FigureWidget()
fig.layout.hovermode = "closest"
fig.layout.clickmode = "event+select"
fig.layout.plot_bgcolor = "white"
fig.layout.paper_bgcolor = "white"

fig.add_scatter(x=sigmas, y=mus, mode="lines", line=dict(color="lightgray"),
                hoverinfo="skip", showlegend=False)

frontier_pts = go.Scatter(
    x=sigmas, y=mus, mode="markers",
    marker=dict(size=7, color="gray", opacity=0.5),
    name="Raw Frontier",
    hoverlabel=dict(bgcolor="white"),
    hovertemplate="σ: %{x:.2%}<br>µ: %{y:.2%}<br>(click to see top weights)<extra></extra>"
)
fig.add_trace(frontier_pts)

fig.add_scatter(x=[voo_sigma_annual], y=[voo_mu_annual], mode="markers",
                marker=dict(symbol="diamond", size=14, color="royalblue", line=dict(width=2, color="black")),
                name="VOO (ref)",
                hoverlabel=dict(bgcolor="white"),
                hovertemplate="VOO\nσ: %{x:.2%}<br>µ: %{y:.2%}<extra></extra>")

mid = len(sigmas) // 2
fig.add_scatter(x=[sigmas[mid]], y=[mus[mid]], mode="markers",
                marker=dict(size=14, color="red", line=dict(width=2, color="black")),
                name="Selected",
                hoverinfo="skip")
sel_idx = len(fig.data) - 1

fig.update_layout(title="Efficient Frontier (Raw Estimates)",
                  xaxis_title="Volatility (σ)", yaxis_title="Expected Return (µ)",
                  height=480)

# --- Top 3 Portfolio Components ---

def format_html(idx: int) -> str:
    w = weights[idx]
    top_idx = np.argsort(w)[-3:][::-1]
    rows = [
        f"<li><b>{symbols[i]}</b> <span style='color:#666'>({name_map.get(symbols[i],'Unknown')})</span>"
        f"<span style='float:right'>{w[i]:.2%}</span></li>"
        for i in top_idx if w[i] > 0.001
    ]
    return (
        "<div style='font-family:Arial, sans-serif; font-size:14px; line-height:1.4; max-width:420px;'>"
        f"<h4 style='margin:4px 0 8px 0; font-size:15px;'>Top 3 ETFs – Frontier Point {idx+1}</h4>"
        "<ul style='list-style:none; padding-left:0; margin:0;'>" + "\n".join(rows) + "</ul></div>"
    )

out = HTML()

# --- Sliders ---

mu_slider = FloatSlider(value=mus[mid], min=mus.min(), max=mus.max(), step=0.0005,
                        description="Target µ", readout_format=".2%", continuous_update=False,
                        layout=dict(width="450px"))

sigma_slider = FloatSlider(value=sigmas[mid], min=sigmas.min(), max=sigmas.max(), step=0.0005,
                           description="Target σ", readout_format=".2%", continuous_update=False,
                           layout=dict(width="450px"))

# --- Sync logic ---

def move_to_index(idx: int):
    with fig.batch_update():
        fig.data[sel_idx].x = [sigmas[idx]]
        fig.data[sel_idx].y = [mus[idx]]
    mu_slider.value = mus[idx]
    sigma_slider.value = sigmas[idx]
    out.value = format_html(idx)

move_to_index(mid)

def on_mu_change(change):
    if change["name"] == "value":
        idx = int(np.argmin(np.abs(mus - change["new"])))
        move_to_index(idx)

def on_sigma_change(change):
    if change["name"] == "value":
        idx = int(np.argmin(np.abs(sigmas - change["new"])))
        move_to_index(idx)

mu_slider.observe(on_mu_change, names="value")
sigma_slider.observe(on_sigma_change, names="value")

fig.data[1].on_click(lambda trace, points, selector: move_to_index(points.point_inds[0]) if points.point_inds else None)

# --- Snap-to button for VOO mu and sigma ---

snap_button = Button(description="Snap to VOO", button_style="info")
snap_criterion = Dropdown(
    options=[("Match µ (return)", "mu"), ("Match σ (volatility)", "sigma")],
    value="mu",
    layout=dict(width="180px")
)

def on_snap(_):
    if snap_criterion.value == "mu":
        idx = int(np.argmin(np.abs(mus - voo_mu_annual)))
    else:
        idx = int(np.argmin(np.abs(sigmas - voo_sigma_annual)))
    move_to_index(idx)

snap_button.on_click(on_snap)

# --- Display ------------------------------------------------------

display(VBox([
    fig,
    mu_slider,
    sigma_slider,
    HBox([snap_button, snap_criterion]),
    out
]))




