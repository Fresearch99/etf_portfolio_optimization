"""
Mean-Variance Optimization (MVO) — one-click Dash UI

This script:
1) Locates the project root (robustly) and loads required CSVs from ./data
2) Cleans/filters the ETF universe
3) Computes a sample-based efficient frontier via CVXPY (long-only, 0–1 box)
4) Builds a small Dash app to explore the frontier, snap to VOO, and display top weights
5) Opens the browser automatically and runs the local server
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard library imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import time
import threading
import webbrowser
from pathlib import Path
from functools import lru_cache
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
# Third-party imports
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, ctx
from flask import request

# ──────────────────────────────────────────────────────────────────────────────
# Global state for the tab-close watchdog 
# ──────────────────────────────────────────────────────────────────────────────
_last_ping = time.time()

# ──────────────────────────────────────────────────────────────────────────────
# Global settings & constants 
# ──────────────────────────────────────────────────────────────────────────────

# For reproducible demos (does not affect optimization, only any random ops if added)
np.random.seed(42)

# Analysis window (years) — not directly used by this file but preserved
ANALYSIS_YEARS = 15

# Optimization & simulation parameters 
FRONTIER_POINTS = 50          # Number of points along the efficient frontier

# Configure pandas display (unchanged)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ──────────────────────────────────────────────────────────────────────────────
# Project root discovery and data path helpers 
# ──────────────────────────────────────────────────────────────────────────────

CANDIDATE_SCRIPT_DIRS = ("scripts", "script")
REQUIRED_DATA_FILES = ("df_etf_metadata.csv", "returns_monthly.csv")


def looks_like_root(p: Path) -> bool:
    """
    Heuristic: a project root must contain a 'data' folder and either
    'scripts' or 'script'. We also require at least one expected CSV
    inside 'data' to avoid false positives.
    """
    if not any((p / d).is_dir() for d in CANDIDATE_SCRIPT_DIRS):
        return False
    data = p / "data"
    if not data.is_dir():
        return False
    return any((data / f).exists() for f in REQUIRED_DATA_FILES)


@lru_cache
def find_project_root() -> Path:
    """
    Try to locate the project root by:
      1) Walking upward from the script directory (or CWD if in a notebook)
      2) Checking each ancestor's immediate children (handles being in a parent dir)
      3) Bounded downward BFS from the CWD
      4) Respecting an explicit override via ETF_OPTIMIZER_ROOT

    Returns
    -------
    Path to the detected project root.

    Raises
    ------
    FileNotFoundError if no plausible root is found.
    """
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()

    # 1) Walk upward
    for p in [start] + list(start.parents):
        if looks_like_root(p):
            return p
        # 2) Check each ancestor's immediate children
        for child in p.iterdir():
            if child.is_dir() and looks_like_root(child):
                return child

    # 3) Bounded downward BFS from CWD (depth ≤ 3)
    max_depth = 3
    q = deque([(Path.cwd().resolve(), 0)])
    visited: set[Path] = set()
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

    # 4) Environment variable override (always wins if valid)
    env = os.getenv("ETF_OPTIMIZER_ROOT")
    if env:
        cand = Path(env).resolve()
        if looks_like_root(cand):
            return cand

    raise FileNotFoundError(f"Could not locate project root from start={start}")


def data_path(name: str) -> Path:
    """Return absolute path to a file inside the ./data folder."""
    return find_project_root() / "data" / name


# ──────────────────────────────────────────────────────────────────────────────
# Data loading 
# ──────────────────────────────────────────────────────────────────────────────

# Metadata (symbols, long names, expense ratios)
df_etf_metadata = pd.read_csv(data_path("df_etf_metadata.csv"))
etf_name_map = dict(zip(df_etf_metadata["Symbol"], df_etf_metadata["Fund name"]))
etf_expense_map = dict(zip(df_etf_metadata["Symbol"], df_etf_metadata["Expense ratio"]))
etf_symbols = list(etf_name_map.keys())

# Returns (monthly, wide format, date index)
returns_monthly = pd.read_csv(data_path("returns_monthly.csv"), index_col=0, parse_dates=True)

# ──────────────────────────────────────────────────────────────────────────────
# Universe filtering: drop sector/overlapping ETFs to keep broad exposures
## ──────────────────────────────────────────────────────────────────────────────

industry_keywords = [
    "Energy", "Health Care", "Consumer", "Materials", "Financials",
    "Utilities", "Real Estate", "Industrials", "Communication", "Information Technology",
]
remove_symbols = ["VGT", "VHT", "VPU", "VDC", "VAW", "VIS", "VFH", "VNQ", "VOX", "VDE", "VCR"]


def is_industry_or_redundant(symbol: str, name_map: dict[str, str]) -> bool:
    """Return True if ETF looks sector-specific or is explicitly excluded."""
    name = name_map.get(symbol, "")
    is_industry = any(keyword in name for keyword in industry_keywords)
    is_redundant = symbol in remove_symbols
    return is_industry or is_redundant


# Filter by sector/redundancy
etf_symbols = [s for s in etf_symbols if not is_industry_or_redundant(s, etf_name_map)]
# Ensure uniqueness (order-preserving)
etf_symbols = list(dict.fromkeys(etf_symbols))
print(f"\nFiltered down to {len(etf_symbols)} ETFs for analysis.")

# Re-load returns (same file) — kept to respect original structure
returns_monthly = pd.read_csv(data_path("returns_monthly.csv"), index_col=0, parse_dates=True)

# ──────────────────────────────────────────────────────────────────────────────
# Data cleaning: drop short histories and any remaining NA rows 
# ──────────────────────────────────────────────────────────────────────────────

MIN_OBSERVATIONS = 10 * 12  # at least 10 years of monthly data
returns_monthly = returns_monthly.dropna(axis=1, thresh=MIN_OBSERVATIONS)
returns_monthly = returns_monthly.dropna(axis=0)

# Final ETF list after cleaning
etf_symbols = returns_monthly.columns.tolist()

# Ensure VOO is present for benchmark comparison
if "VOO" not in etf_symbols:
    raise ValueError("VOO data is missing or was dropped. Required for benchmark comparison.")

# Expense ratios aligned to final symbol order
expense_vector = np.array([etf_expense_map.get(sym, 0.0) for sym in etf_symbols])

print(f"\nFinal analysis will use {len(etf_symbols)} ETFs over {len(returns_monthly)} months.")
print(f"Analysis period: {returns_monthly.index.min().date()} → {returns_monthly.index.max().date()}")

# ──────────────────────────────────────────────────────────────────────────────
# Summary stats used by MVO 
# ──────────────────────────────────────────────────────────────────────────────

# Annualized mean returns (net of expense ratios)
annual_mu_sample = (returns_monthly.mean().values * 12) - expense_vector

# Annualized covariance matrix
sample_cov = returns_monthly.cov().values
annual_cov_sample = sample_cov * 12

# VOO benchmark stats (annualized)
voo_returns_monthly = returns_monthly["VOO"]
voo_mu_annual = voo_returns_monthly.mean() * 12 - etf_expense_map.get("VOO", 0.0)
voo_sigma_annual = voo_returns_monthly.std() * np.sqrt(12)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: pick portfolio closest to a target µ or σ 
# ──────────────────────────────────────────────────────────────────────────────

def select_portfolio(frontier: dict, target_metric: str, target_value: float) -> tuple[int | None, np.ndarray | None]:
    """
    Select the frontier portfolio closest to a target (by absolute distance).

    Parameters
    ----------
    frontier : dict
        Dict with lists 'mu', 'sigma', 'weights'.
    target_metric : {'mu','sigma'}
        Which series to match against.
    target_value : float
        Target expected return or volatility (annualized).

    Returns
    -------
    (index, weights) or (None, None) if the frontier list is empty.
    """
    if not frontier[target_metric]:
        return None, None
    diffs = np.abs(np.array(frontier[target_metric]) - target_value)
    idx = int(diffs.argmin())
    return idx, frontier["weights"][idx]


# ──────────────────────────────────────────────────────────────────────────────
# Efficient frontier via CVXPY 
# ──────────────────────────────────────────────────────────────────────────────

def efficient_frontier(
    covariance_matrix, expected_returns, n_points: int = 50, lambda_l1: float = 0.0
) -> dict:
    """
    Compute the (pruned) efficient frontier under long-only box constraints.

    Objective:
        minimize   w' Σ w + λ ||w||_1
        subject to sum(w) = 1,   0 ≤ w ≤ 1,    μ'w == target   (or relaxed to ≥)

    Notes
    -----
    - Uses OSQP first (fast for QP); falls back to SCS if needed.
    - If equality µ'w == target is infeasible, retries with µ'w ≥ target.
    - Adds a tiny ridge to Σ during the solve for numerical stability,
      but reports σ using the original (un-ridged) Σ.
    """
    # Inputs → numpy
    Sigma_in = np.asarray(covariance_matrix, dtype=float)
    mu = np.asarray(expected_returns, dtype=float).ravel()
    n = mu.shape[0]

    # Symmetrize + tiny ridge (solver only)
    Sigma = 0.5 * (Sigma_in + Sigma_in.T)
    Sigma = Sigma + 1e-10 * np.eye(n)

    # Target return grid across the observed range
    target_mus = np.linspace(mu.min(), mu.max(), n_points)

    frontier = {"mu": [], "sigma": [], "weights": []}

    for target in target_mus:
        w = cp.Variable(n)

        # Base constraints: long-only, fully invested
        cons_eq = [cp.sum(w) == 1, w >= 0, w <= 1, mu @ w == target]
        cons_ge = [cp.sum(w) == 1, w >= 0, w <= 1, mu @ w >= target]

        obj = cp.Minimize(cp.quad_form(w, Sigma) + lambda_l1 * cp.norm1(w))

        # Try equality; if infeasible, relax to ≥
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
            # Skip this target if still infeasible
            continue

        # Clean & renormalize weights
        w_sol = np.asarray(w.value).ravel()
        w_sol[np.abs(w_sol) < 1e-10] = 0.0
        s = w_sol.sum()
        w_sol = (w_sol / s) if s > 0 else np.ones(n) / n

        # Report σ with original Σ (without ridge)
        sigma = float(np.sqrt(w_sol @ Sigma_in @ w_sol))
        mu_realized = float(w_sol @ mu)

        frontier["mu"].append(mu_realized)
        frontier["sigma"].append(sigma)
        frontier["weights"].append(w_sol)

    return frontier


# ──────────────────────────────────────────────────────────────────────────────
# Frontier pruning: keep the efficient (upper) arm only 
# ──────────────────────────────────────────────────────────────────────────────

def prune_frontier(frontier: dict) -> dict:
    """
    Remove dominated points: sort by σ ascending and keep only
    strictly increasing µ as σ increases.
    """
    vol = np.asarray(frontier["sigma"])
    ret = np.asarray(frontier["mu"])
    wlist = list(frontier["weights"])

    # 1) Sort by volatility
    order = np.argsort(vol)
    vol, ret = vol[order], ret[order]
    wlist = [wlist[i] for i in order]

    # 2) Keep strictly increasing returns
    keep_idx = []
    last_best_ret = -np.inf
    for i in range(len(ret)):
        if ret[i] > last_best_ret + 1e-12:
            keep_idx.append(i)
            last_best_ret = ret[i]

    return {
        "sigma": vol[keep_idx].tolist(),
        "mu": ret[keep_idx].tolist(),
        "weights": [wlist[i] for i in keep_idx],
    }


# Compute/prune the sample frontier (unchanged)
ef_raw = prune_frontier(
    efficient_frontier(annual_cov_sample, annual_mu_sample, n_points=FRONTIER_POINTS)
)

# Match VOO’s µ or σ on the frontier (weights not directly used in UI, preserved)
_, w_mu_raw = select_portfolio(ef_raw, "mu", voo_mu_annual)
_, w_sigma_raw = select_portfolio(ef_raw, "sigma", voo_sigma_annual)

# Ensure arrays exist for plotting/interaction
sigmas = np.asarray(ef_raw["sigma"])
mus = np.asarray(ef_raw["mu"])
weights = np.asarray(ef_raw["weights"])

# Expose symbols/name_map (preserving original globals fallback)
symbols = globals().get("symbols", etf_symbols)
name_map = globals().get("name_map", etf_name_map)

# Slider resolution in absolute units (0.0001 ≈ 1 bp)
RESOLUTION = 0.0001

# ──────────────────────────────────────────────────────────────────────────────
# Plotly figure — static parts 
# ──────────────────────────────────────────────────────────────────────────────

def build_initial_fig() -> go.Figure:
    """Create the base figure: frontier line + points, VOO marker, selected marker."""
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

    # Frontier (line)
    fig.add_scatter(
        x=sigmas,
        y=mus,
        mode="lines",
        line=dict(color="lightgray"),
        hoverinfo="skip",
        showlegend=False,
    )

    # Frontier points
    fig.add_scatter(
        x=sigmas,
        y=mus,
        mode="markers",
        marker=dict(size=7, color="gray", opacity=0.5),
        name="Raw Frontier",
        hoverlabel=dict(bgcolor="white"),
        hovertemplate="σ: %{x:.2%}<br>µ: %{y:.2%}<br>(click to see top weights)<extra></extra>",
    )

    # VOO reference
    fig.add_scatter(
        x=[voo_sigma_annual],
        y=[voo_mu_annual],
        mode="markers",
        marker=dict(symbol="diamond", size=14, color="royalblue", line=dict(width=2, color="black")),
        name="VOO (ref)",
        hoverlabel=dict(bgcolor="white"),
        hovertemplate="VOO<br>σ: %{x:.2%}<br>µ: %{y:.2%}<extra></extra>",
    )

    # Selected point (initialized to mid)
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


# ──────────────────────────────────────────────────────────────────────────────
# Small UI helpers 
# ──────────────────────────────────────────────────────────────────────────────

def top3_component(idx: int) -> html.Div:
    """Render a small list of the top-3 ETF weights for a frontier point."""
    w = weights[idx]
    top_idx = np.argsort(w)[-3:][::-1]

    items = [
        html.Li(
            [
                html.B(symbols[i]),
                html.Span(f" ({name_map.get(symbols[i], 'Unknown')})", style={"color": "#666"}),
                html.Span(f"{w[i]:.2%}", style={"float": "right"}),
            ]
        )
        for i in top_idx
        if w[i] > 0.001
    ]

    return html.Div(
        [
            html.H4(
                f"Top 3 ETFs – Frontier Point {idx + 1}",
                style={"margin": "4px 0 8px 0", "fontSize": "15px"},
            ),
            html.Ul(items, style={"listStyle": "none", "paddingLeft": "0", "margin": "0"}),
        ],
        style={"fontFamily": "Arial, sans-serif", "fontSize": "14px", "lineHeight": "1.4", "maxWidth": "420px"},
    )


def closest_point(target_mu: float, target_sigma: float) -> int:
    """Index of the frontier point nearest to (target_sigma, target_mu)."""
    dist = (mus - target_mu) ** 2 + (sigmas - target_sigma) ** 2
    return int(np.argmin(dist))


# ──────────────────────────────────────────────────────────────────────────────
# Dash app: routes, watchdog, layout 
# ──────────────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
fig_initial = build_initial_fig()
mid_idx = len(sigmas) // 2  # initial selection

# Shutdown endpoint so the app can stop via browser tab close
@app.server.route("/shutdown", methods=["POST"])
def _shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()       # dev server path
    else:
        os._exit(0)  # fallback
    return "Shutting down"


# Heartbeat endpoint (browser pings periodically)
@app.server.route("/ping", methods=["POST"])
def _ping():
    global _last_ping
    _last_ping = time.time()
    return "ok"


def _watchdog(timeout: int = 20) -> None:
    """
    Watchdog thread: if the browser stops pinging for `timeout` seconds,
    exit the process. Helps auto-stop on tab close in edge cases.
    """
    global _last_ping
    while True:
        time.sleep(5)
        if time.time() - _last_ping > timeout:
            os._exit(0)


# Start watchdog thread (daemon)
threading.Thread(target=_watchdog, args=(20,), daemon=True).start()

# Layout with graph, two sliders, "Snap to VOO" button, and top-3 weights box
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

        # Inject small script: ping server every 5s; attempt shutdown on tab close
        html.Script(
            """
            (function(){
              const post = (u,b) => {
                try {
                  if (navigator.sendBeacon) return navigator.sendBeacon(u, new Blob([b||'.'], {type:'text/plain'}));
                  fetch(u, {method:'POST', keepalive:true, body:b||'.'});
                } catch(e){}
              };
              const base = window.location.origin;
              // Heartbeat every 5s
              const t = setInterval(()=>post(base + '/ping'), 5000);
              // Attempt to shut down immediately when the tab/window goes away
              const bye = () => { try { post(base + '/shutdown','bye'); } catch(e){} };
              window.addEventListener('pagehide', bye);
              window.addEventListener('beforeunload', bye);
            })();
            """
        ),
    ],
    style={"width": "700px", "margin": "auto"},
)

# ──────────────────────────────────────────────────────────────────────────────
# Dash callbacks 
# ──────────────────────────────────────────────────────────────────────────────

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
    """
    Update the selected point, sliders, and top-weights view based on:
      - click on a frontier point,
      - "Snap to VOO" press (by µ or σ),
      - or nearest point to the (µ, σ) sliders.
    """
    trigger = ctx.triggered_id
    sel_idx = mid_idx

    if trigger == "frontier-graph" and click_data:
        sel_idx = click_data["points"][0]["pointIndex"]
    elif trigger == "snap-btn" and n_clicks:
        sel_idx = int(np.argmin(np.abs(mus - voo_mu_annual))) if snap_choice == "mu" \
                  else int(np.argmin(np.abs(sigmas - voo_sigma_annual)))
    else:
        sel_idx = closest_point(mu_val, sigma_val)

    # Rebuild figure and move the "Selected" marker (trace index 3)
    fig = build_initial_fig()
    fig.data[3].x = [sigmas[sel_idx]]
    fig.data[3].y = [mus[sel_idx]]

    # Slider feedback rounded to two decimals
    mu_val_new = round(float(mus[sel_idx]), 2)
    sigma_val_new = round(float(sigmas[sel_idx]), 2)

    html_output = top3_component(sel_idx)
    return fig, mu_val_new, sigma_val_new, html_output


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint 
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Open the browser shortly after server starts
    threading.Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:8050/")).start()

    # Dash 3.x API (no reloader to avoid opening two tabs)
    app.run(debug=True, port=8050, use_reloader=False)
