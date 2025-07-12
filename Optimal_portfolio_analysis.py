# ==============================================================================
#
#               PORTFOLIO OPTIMIZATION & DYNAMIC ASSET ALLOCATION
#
# ==============================================================================
#
# OVERVIEW:
#
# This script performs a comprehensive portfolio analysis using various optimization
# techniques. It starts with classic Mean-Variance Optimization and then explores
# more advanced methods to create robust and dynamic asset allocation strategies.
#
# The script is divided into the following main parts:
#
#   1.  SETUP & CONFIGURATION:
#       -   Loads necessary libraries and defines global settings.
#
#   2.  DATA LOADING & PREPARATION:
#       -   Fetches a list of Vanguard ETFs and their metadata.
#       -   Downloads historical price data for these ETFs from Yahoo Finance.
#       -   Processes the data to calculate monthly returns over a 10-year window.
#
#   3.  STATIC PORTFOLIO ANALYSIS (MEAN-VARIANCE OPTIMIZATION):
#       -   Calculates the expected returns (mu) and covariance matrix of the ETFs.
#       -   Introduces "shrinkage" techniques (Ledoit-Wolf) to create more
#           stable estimates of these parameters.
#       -   Defines a function to compute the "Efficient Frontier," which represents
#           the set of optimal portfolios.
#       -   Constructs several static portfolios and compares them to a benchmark (VOO).
#
#   4.  ADVANCED STATIC MODELS & ROBUSTNESS CHECKS:
#       -   Performs a Monte Carlo simulation to forecast portfolio performance.
#       -   Uses a "Resampled Efficient Frontier" (bootstrapping) to build a
#           portfolio that is less sensitive to estimation errors.
#       -   Implements a "Rolling Window" analysis to see how the optimal
#           portfolio weights would have changed over time.
#       -   Implement "Black-Litterman and Hierarchical Risk Parity (HRP)"
#           analysis additional advanced methods.
#
#   5.  REGIME-SWITCHING MODEL:
#       -   Loads external economic data (VIX, US Treasury yields) that may
#           influence market behavior.
#       -   Fits a Markov Regime-Switching model to identify distinct market
#           "regimes" (e.g., low-volatility growth, high-volatility decline).
#
#   6.  REGIME-AWARE DYNAMIC STRATEGY:
#       -   Calculates an optimal portfolio for each identified market regime.
#       -   Backtests a dynamic strategy that adjusts its portfolio allocation
#           based on the real-time probability of being in each regime.
#
#   7.  FINAL PERFORMANCE COMPARISON:
#       -   Plots the cumulative returns of all tested strategies.
#       -   Calculates and displays a table of key performance metrics (Return,
#           Volatility, Sharpe Ratio, Max Drawdown) for a comprehensive comparison.
#
# ==============================================================================

# ==============================================================================
# SECTION 1: SETUP & CONFIGURATION
# ==============================================================================

# --- Standard Library Imports ---
import os

# --- Third-Party Library Imports ---
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import cvxopt as opt
import statsmodels.api as sm

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
import cvxpy as cp

from sklearn.covariance import LedoitWolf
from pandas_datareader.data import DataReader
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Global Settings & Constants ---

# NOTE: You may need to change this to the directory where your script and data file are located.
DIRECTORY = '.'
# The file 'vanguard_etf_list.xlsx' is expected to be in the above directory.
# It should contain ETF metadata like Symbol, Fund name, and Expense ratio.

# Analysis Period
ANALYSIS_YEARS = 15

# Optimization & Simulation Parameters
FRONTIER_POINTS = 50          # Number of points to calculate on the efficient frontier.
MC_SIM_SCENARIOS = 10000      # Number of scenarios for Monte Carlo simulation.
MC_SIM_HORIZON_MONTHS = 120   # 10-year horizon for simulation.
RESAMPLE_ITERATIONS = 100     # Number of bootstrap iterations for resampled frontier.
ROLLING_WINDOW_MONTHS = 60    # 5-year rolling window for dynamic weight analysis.

# Regime Modeling Parameters
MIN_OBS_PER_REGIME = 10       # Minimum data points required to consider a regime valid.
MAX_REGIMES_TO_TEST = 4       # Test models with 2 up to this number of regimes.

# --- Initial Setup ---
os.chdir(DIRECTORY)

# Configure pandas to display floating-point numbers with 3 decimal places.
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Configure the CVXOPT solver to not display progress messages during optimization.
opt.solvers.options['show_progress'] = False


# ==============================================================================
# SECTION 2: DATA LOADING & PREPARATION
# ==============================================================================
print("--- Section 2: Loading ETF Data and Calculating Returns ---")

# --- 2a. Load ETF Metadata ---
# We attempt to load ETF metadata from a local Excel file.
# This file provides full fund names and expense ratios for our list of ETFs.
# If the file is not found, we fall back to a predefined list of symbols.
try:
    etf_lookup_df = pd.read_excel("vanguard_etf_list.xlsx", skiprows=4)
    etf_name_map = dict(zip(etf_lookup_df['Symbol'], etf_lookup_df['Fund name']))
    etf_expense_map = dict(zip(etf_lookup_df['Symbol'], etf_lookup_df['Expense ratio']))
    # Clean the dictionaries by removing any entries where the symbol is NaN (Not a Number).
    etf_name_map = {k: v for k, v in etf_name_map.items() if pd.notna(k)}
    etf_expense_map = {k: v for k, v in etf_expense_map.items() if pd.notna(k)}
    etf_symbols = list(etf_name_map.keys())
    print(f"Successfully loaded metadata for {len(etf_symbols)} ETFs.")
except FileNotFoundError:
    print("Warning: 'vanguard_etf_list.xlsx' not found.")
    print("Using a smaller, predefined list of ETFs as a fallback.")
    etf_symbols = ['VOO', 'VTI', 'VEA', 'VWO', 'BND', 'BNDX', 'VGIT', 'VGLT', 'VTIP', 'MUB']
    # Use placeholder names and a generic expense ratio if the file is missing.
    etf_name_map = {s: s for s in etf_symbols}
    etf_expense_map = {s: 0.0003 for s in etf_symbols} # 0.03%

# --- 2b. Filter the ETF Universe ---
# To create a diversified portfolio of broad asset classes, we remove specialized
# sector-specific ETFs and redundant funds.
industry_keywords = [
    'Energy', 'Health Care', 'Consumer', 'Materials', 'Financials',
    'Utilities', 'Real Estate', 'Industrials', 'Communication', 'Information Technology'
]
# List of specific ETFs to remove, often because they are sector-focused or
# overlap significantly with broader ETFs like VTI.
remove_symbols = ['VGT', 'VHT', 'VPU', 'VDC', 'VAW', 'VIS', 'VFH', 'VNQ', 'VOX', 'VDE', 'VCR']

def is_industry_or_duplicate(symbol):
    """Checks if an ETF is a sector-specific or redundant fund based on its name or symbol."""
    name = etf_name_map.get(symbol, '')
    is_industry = any(keyword in name for keyword in industry_keywords)
    is_duplicate = symbol in remove_symbols
    return is_industry or is_duplicate

# Apply the filter and remove any duplicate symbols that might exist in the original list.
etf_symbols = [sym for sym in etf_symbols if not is_industry_or_duplicate(sym)]
etf_symbols = list(dict.fromkeys(etf_symbols))
print(f"Filtered down to {len(etf_symbols)} ETFs for analysis.")


# --- 2c. Fetch Historical Price Data ---
def get_total_return_series(ticker):
    """
    Fetches the maximum available historical price data for a given ticker
    from Yahoo Finance.

    Args:
        ticker (str): The stock or ETF symbol.

    Returns:
        pd.DataFrame: A DataFrame with a single column of historical closing prices,
                      adjusted for dividends and splits. Returns an empty DataFrame
                      if the ticker cannot be fetched.
    """
    print(f"Processing {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        # 'back_adjust=True' provides a total return series by adjusting prices
        # for both dividends and stock splits. 'auto_adjust=False' is needed
        # for back_adjust to work.
        df = stock.history(
            period="max",
            auto_adjust=False,
            back_adjust=True
        )[['Close']].rename(columns={'Close': ticker})
        return df
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        return pd.DataFrame()

# Loop through the filtered list of ETFs and combine their price series into one DataFrame.
all_prices = pd.DataFrame()
for ticker in etf_symbols:
    price_df = get_total_return_series(ticker)
    if not price_df.empty:
        all_prices = pd.concat([all_prices, price_df], axis=1)

# --- 2d. Process and Clean Return Data ---
# Standardize the index to datetime objects without timezone information.
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)

# Resample daily prices to month-end prices, then calculate monthly percentage returns.
returns_monthly = all_prices.resample('M').last().pct_change()

# Limit the data to the last N years for a more relevant analysis window.
cutoff_date = returns_monthly.index.max() - pd.DateOffset(years=ANALYSIS_YEARS)
returns_monthly = returns_monthly[returns_monthly.index >= cutoff_date]

# Data Cleaning:
# 1. Drop any ETF (column) that doesn't have at least 50% of the data points in our window.
min_observations = int(len(returns_monthly) * 0.50)
returns_monthly = returns_monthly.dropna(axis=1, thresh=min_observations)

# 2. Drop any month (row) that still has missing values after filtering columns.
returns_monthly = returns_monthly.dropna(axis=0)

# Update the final list of ETF symbols and related data based on the cleaned DataFrame.
etf_symbols = returns_monthly.columns.tolist()

# The S&P 500 ETF (VOO) is used as our primary benchmark. The analysis cannot
# proceed without it.
if 'VOO' not in etf_symbols:
    raise ValueError("VOO data is missing or was dropped. It is required for the benchmark comparison.")

# Create a NumPy array of expense ratios in the same order as our final ETF symbols.
# This will be used to calculate net returns.
expense_vector = np.array([etf_expense_map.get(sym, 0.0) for sym in etf_symbols])
print(f"Final analysis will use {len(etf_symbols)} ETFs over {len(returns_monthly)} months.")


# ==============================================================================
# SECTION 3: STATIC PORTFOLIO ANALYSIS (MEAN-VARIANCE OPTIMIZATION)
# ==============================================================================
print("\n--- Section 3: Performing Static Mean-Variance Optimization ---")

# --- 3a. Estimate Expected Returns and Covariance ---
# Portfolio optimization requires two key inputs:
# 1. Expected Returns (mu): The anticipated return for each asset.
# 2. Covariance Matrix: A measure of how asset returns move together.

# Calculate historical annualized mean returns, net of expense ratios.
# We multiply by 12 to annualize the monthly mean returns.
annual_mu_sample = returns_monthly.mean().values * 12 - expense_vector

# The sample covariance matrix is calculated from the historical returns.
sample_cov = returns_monthly.cov().values
annual_cov_sample = sample_cov * 12

# --- 3b. Apply Shrinkage to Improve Estimates ---
# Historical sample estimates can be "noisy" and may not be good predictors of the future.
# Shrinkage techniques adjust these estimates to be more stable and robust.

# Ledoit-Wolf Shrinkage for the Covariance Matrix:
# This method computes an optimal blend of the sample covariance matrix and a more
# structured, less noisy target matrix. It's a standard way to get a more reliable
# covariance estimate, especially when the number of assets is large relative to
# the number of observations.
lw = LedoitWolf().fit(returns_monthly.values)
annual_cov_shrunk = lw.covariance_ * 12

# Note on James-Stein Shrinkage for Mean Returns:
# The original script attempted James-Stein shrinkage for the mean returns.
# However, this method can be overly aggressive, often shrinking all expected
# returns towards the grand mean, which eliminates valuable differences between assets.
# We will proceed using the simple historical sample means, as they are more common
# in practice for this type of analysis.
#
# def james_stein_shrinkage(mu):
#     mu_bar = mu.mean()
#     n = len(mu)
#     # Formula corrected for clarity
#     shrinkage_factor = 1 - ((n - 3) * mu.var()) / ((n - 1) * np.sum((mu - mu_bar) ** 2))
#     shrinkage_factor = max(0, min(shrinkage_factor, 1))
#     return shrinkage_factor * mu_bar + (1 - shrinkage_factor) * mu
# annual_mu_shrunk = james_stein_shrinkage(annual_mu_sample)
#
# Based on the original script's conclusion ("very aggressive shrinkage... not useful"),
# we will use the un-shrunk `annual_mu_sample` for our primary analysis.
annual_mu = annual_mu_sample

print("Annualized Expected Returns (Sample):")
print(pd.Series(annual_mu, index=etf_symbols).round(3))


# --- 3c. Define the Efficient Frontier Optimizer ---
def efficient_frontier(cov_mat, mu_vec, n_points=50):
    """
    Calculates the efficient frontier using the Markowitz mean-variance optimization model.

    The "Efficient Frontier" is the set of portfolios that provide the highest
    expected return for a given level of risk (volatility).

    This function uses the CVXOPT quadratic programming solver to find the portfolio
    weights that minimize portfolio variance for a range of target expected returns.

    Args:
        cov_mat (np.array): The annualized covariance matrix of asset returns.
        mu_vec (np.array): The annualized vector of asset expected returns.
        n_points (int): The number of points to calculate along the frontier.

    Returns:
        dict: A dictionary containing the returns ('mu'), volatilities ('sigma'),
              and portfolio weights ('weights') for each point on the frontier.
    """
    n = len(mu_vec)  # Number of assets

    # The optimization problem is to minimize: (1/2) * w' * P * w
    # subject to constraints. Here, P is the covariance matrix.
    P = opt.matrix(cov_mat)
    # The 'q' term is for a linear part of the objective, which is zero here.
    q = opt.matrix(np.zeros((n, 1)))

    # Constraint 1: Weights must be non-negative (G*w <= h).
    # -w_i <= 0  (or w_i >= 0)
    # Constraint 2: Weights must be less than 1 (w_i <= 1).
    G = opt.matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = opt.matrix(np.vstack([np.zeros((n, 1)), np.ones((n, 1))]))

    # Constraint 3: Sum of weights must equal 1, and portfolio return must equal target. (A*w = b)
    # The solver will iterate through different target returns (`mu_target`).
    A = opt.matrix(np.vstack([mu_vec, np.ones((1, n))]))

    # Iterate through a range of target returns to trace the frontier.
    target_mus = np.linspace(mu_vec.min(), mu_vec.max(), n_points)
    frontier = {'mu': [], 'sigma': [], 'weights': []}

    for mu_target in target_mus:
        b = opt.matrix([mu_target, 1.0])  # Target return and sum of weights = 1
        try:
            solution = opt.solvers.qp(P, q, G, h, A, b)
            if solution['status'] == 'optimal':
                weights = np.array(solution['x']).flatten()
                # Calculate the resulting portfolio volatility (sigma).
                sigma = np.sqrt(weights.T @ cov_mat @ weights)
                frontier['mu'].append(mu_target)
                frontier['sigma'].append(sigma)
                frontier['weights'].append(weights)
        except ValueError:
            # The solver may fail for some target returns if no solution exists.
            pass

    return frontier

# --- 3d. Define Benchmark and Helper Functions ---
voo_returns_monthly = returns_monthly['VOO']
voo_mu_annual = voo_returns_monthly.mean() * 12 - etf_expense_map.get('VOO', 0.0)
voo_sigma_annual = voo_returns_monthly.std() * np.sqrt(12)

def select_portfolio(frontier, target_metric, target_value):
    """
    Selects a portfolio from the efficient frontier that is closest to a target value.

    Args:
        frontier (dict): The efficient frontier dictionary.
        target_metric (str): The metric to match ('mu' or 'sigma').
        target_value (float): The target return or volatility.

    Returns:
        tuple: The index and weights of the selected portfolio, or (None, None).
    """
    if not frontier[target_metric]:
        return None, None
    diffs = np.abs(np.array(frontier[target_metric]) - target_value)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]

# --- 3e. Generate Frontiers and Select Portfolios ---
print("Generating efficient frontiers...")
# Frontier using simple sample estimates
ef_raw = efficient_frontier(annual_cov_sample, annual_mu, n_points=FRONTIER_POINTS)
# Frontier using shrinkage-adjusted covariance
ef_shrunk = efficient_frontier(annual_cov_shrunk, annual_mu, n_points=FRONTIER_POINTS)

# Find portfolios on each frontier that match the VOO benchmark's risk or return
_, w_mu_raw = select_portfolio(ef_raw, 'mu', voo_mu_annual)
_, w_sigma_raw = select_portfolio(ef_raw, 'sigma', voo_sigma_annual)
_, w_mu_shrunk = select_portfolio(ef_shrunk, 'mu', voo_mu_annual)
_, w_sigma_shrunk = select_portfolio(ef_shrunk, 'sigma', voo_sigma_annual)

# Display the composition of the selected portfolios
portfolios_to_display = {
    'Raw (Mu-matched)': w_mu_raw,
    'Raw (Sigma-matched)': w_sigma_raw,
    'Shrunk (Mu-matched)': w_mu_shrunk,
    'Shrunk (Sigma-matched)': w_sigma_shrunk
}

for label, weights in portfolios_to_display.items():
    if weights is not None:
        top_indices = np.argsort(weights)[-3:][::-1]
        print(f"\nTop 3 ETFs for {label} Portfolio:")
        for i in top_indices:
            if weights[i] > 0.001: # Only show assets with meaningful weight
                symbol = etf_symbols[i]
                name = etf_name_map.get(symbol, 'Unknown')
                print(f"  {symbol} ({name}): {weights[i]:.2%}")

# --- 3f. Plot Static Efficient Frontiers ---
plt.figure(figsize=(10, 7))
plt.plot(ef_raw['sigma'], ef_raw['mu'], 'o-', label='Raw Estimate Frontier', alpha=0.7)
plt.plot(ef_shrunk['sigma'], ef_shrunk['mu'], 'o-', label='Shrunk Covariance Frontier', lw=2)
plt.scatter([voo_sigma_annual], [voo_mu_annual], color='red', marker='X', s=200, label='VOO Benchmark', zorder=5)

plt.title('Efficient Frontiers: Raw vs. Shrunk Covariance')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Expected Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================================================================
# SECTION 4: ADVANCED STATIC MODELS & ROBUSTNESS CHECKS
# ==============================================================================
print("\n--- Section 4: Performing Robustness Checks ---")

# --- 4a. Monte Carlo Simulation ---
# We simulate future returns to see how our portfolios might perform under a wide
# range of possible outcomes, based on the historical return distribution.
print(f"Running Monte Carlo simulation with {MC_SIM_SCENARIOS} scenarios...")

# Use monthly parameters for the simulation
monthly_mu_sample = annual_mu / 12

rng = np.random.default_rng(seed=42)
simulated_returns_monthly = rng.multivariate_normal(
    mean=monthly_mu_sample,
    cov=sample_cov,
    size=(MC_SIM_SCENARIOS, MC_SIM_HORIZON_MONTHS)
)

def simulate_performance(weights):
    """Calculates performance metrics from simulated returns."""
    if weights is None:
        return {'mean': np.nan, 'vol': np.nan, 'VaR_5': np.nan}
    # Calculate portfolio returns for each scenario and time step.
    portfolio_sim_returns = simulated_returns_monthly @ weights
    # Annualize the results
    annual_mean_return = np.mean(portfolio_sim_returns) * 12
    annual_volatility = np.std(portfolio_sim_returns) * np.sqrt(12)
    # Value-at-Risk (VaR): The worst expected loss at a 5% confidence level.
    var_5_percent = np.percentile(portfolio_sim_returns, 5) * 12
    return {
        'mean': annual_mean_return,
        'vol': annual_volatility,
        'VaR_5': var_5_percent
    }

# Compare simulated performance of Sigma-matched portfolios vs. the benchmark
perf_sigma = {
    'Raw': simulate_performance(w_sigma_raw),
    'Shrunk': simulate_performance(w_sigma_shrunk),
    'VOO': {'mean': voo_mu_annual, 'vol': voo_sigma_annual, 'VaR_5': np.nan} # VOO is the baseline
}
print("\nSimulated Performance Summary (Sigma-matched to VOO):")
print(pd.DataFrame(perf_sigma).T)


# --- 4b. Resampled Efficient Frontier (Bootstrapping) ---
# This technique addresses "estimation error" by creating many new return datasets
# via bootstrapping (sampling with replacement). An optimal portfolio is found for
# each bootstrapped sample, and the final portfolio is the average of all these
# optimal portfolios. This leads to a more diversified and stable allocation.
print(f"\nRunning Resampled Frontier simulation with {RESAMPLE_ITERATIONS} iterations...")

n_obs = returns_monthly.shape[0]
resampled_weights_list = []

for i in range(RESAMPLE_ITERATIONS):
    if (i + 1) % 25 == 0:
        print(f"  Resample iteration {i+1}/{RESAMPLE_ITERATIONS}...")

    # Create a bootstrap sample of the monthly returns.
    boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
    returns_boot = returns_monthly.iloc[boot_indices]

    # Recalculate parameters for the bootstrap sample.
    mu_boot = returns_boot.mean().values * 12 - expense_vector
    try:
        # Use shrunk covariance for better stability in each sample.
        lw_boot = LedoitWolf().fit(returns_boot.values)
        cov_boot = lw_boot.covariance_ * 12

        # Generate frontier and select the sigma-matched portfolio.
        ef_boot = efficient_frontier(cov_boot, mu_boot, n_points=30)
        _, w_boot = select_portfolio(ef_boot, 'sigma', voo_sigma_annual)

        if w_boot is not None:
            resampled_weights_list.append(w_boot)
    except (ValueError, np.linalg.LinAlgError):
        # Skip iteration if solver or covariance estimation fails.
        continue

# The final resampled portfolio is the average of the weights from all iterations.
if resampled_weights_list:
    w_resampled = np.mean(resampled_weights_list, axis=0)
    top_indices = np.argsort(w_resampled)[-5:][::-1]
    print("\nTop 5 ETFs for Resampled Portfolio:")
    for i in top_indices:
        symbol = etf_symbols[i]
        name = etf_name_map.get(symbol, 'Unknown')
        print(f"  {symbol} ({name}): {w_resampled[i]:.2%}")
else:
    w_resampled = None
    print("\nCould not generate a resampled portfolio.")


# --- 4c. Rolling Window Estimation ---
# This analysis shows how the optimal portfolio allocation would have changed over time
# as new data became available, providing insight into the strategy's stability.
print(f"\nPerforming Rolling Window analysis with a {ROLLING_WINDOW_MONTHS}-month window...")

rolling_dates = returns_monthly.index[ROLLING_WINDOW_MONTHS:]
rolling_weights_list = []

for date in rolling_dates:
    # Create a data window of the last N months.
    window_data = returns_monthly.loc[:date].iloc[-ROLLING_WINDOW_MONTHS:]

    # Estimate parameters on the window.
    mu_roll = window_data.mean().values * 12 - expense_vector
    
    # Use shrunk covariance for better performance with smaller sample.
    lw_roll = LedoitWolf().fit(window_data.values)
    cov_roll = lw_roll.covariance_ * 12

    try:
        # Find the optimal (sigma-matched) portfolio for this period.
        ef_roll = efficient_frontier(cov_roll, mu_roll, n_points=30)
        _, w_roll = select_portfolio(ef_roll, 'sigma', voo_sigma_annual)
        if w_roll is not None:
            rolling_weights_list.append(pd.Series(w_roll, index=etf_symbols, name=date))
    except (ValueError, np.linalg.LinAlgError):
        continue

# Combine results and plot the weight changes for the most important assets.
if rolling_weights_list:
    rolling_weights_df = pd.concat(rolling_weights_list, axis=1).T
    top_etfs = rolling_weights_df.mean().sort_values(ascending=False).head(5).index

    rolling_weights_df[top_etfs].plot(
        figsize=(12, 7),
        title='Top 5 ETF Weights Over Time (Rolling Optimization)'
    )
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()


# --- 4d. Black-Litterman  ---
# This implementation uses a data-driven prior: a shrinkage estimator that blends
# the sample mean and an equal-mean neutral vector. The investor expresses a single
# view that VEA will outperform VWO by 1% annualized, which is encoded via matrix P and vector Q.
# The posterior expected returns m_bl are then used in a constrained mean-variance optimizer.
print("\n--- Black-Litterman Portfolio Optimization ---")

# 1. Estimate a data-driven prior (shrunk sample mean)
neutral_mean = np.full_like(annual_mu_sample, annual_mu_sample.mean())
lambda_ = 0.2  # Shrinkage intensity
pi = lambda_ * annual_mu_sample + (1 - lambda_) * neutral_mean


# 2. Specify investor views (e.g., tilt toward international or small-cap)
# For illustration, assume view: international (VEA, VWO) will outperform
P = np.array([[0, 0, 0, 0, 1, -1]])  # VEA - VWO
Q = np.array([0.01])  # View: VEA outperform VWO by 1% annualized

# View uncertainty
omega = np.diag(np.full(len(Q), 0.0025))  # Moderate confidence
bl_tau = 0.05

# Black-Litterman posterior mean
M_inverse = np.linalg.inv(np.linalg.inv(bl_tau * annual_cov_shrunk) + P.T @ np.linalg.inv(omega) @ P)
m_bl = M_inverse @ (np.linalg.inv(bl_tau * annual_cov_shrunk) @ pi + P.T @ np.linalg.inv(omega) @ Q)

# Optimize using mean-variance
w_bl = cp.Variable(len(etf_symbols))
objective = cp.Maximize(m_bl @ w_bl - cp.quad_form(w_bl, annual_cov_shrunk) * 0.5)
constraints = [cp.sum(w_bl) == 1, w_bl >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()
w_bl_opt = w_bl.value

print("Optimal Black-Litterman Portfolio Weights:")
for i, wt in enumerate(w_bl_opt):
    if wt > 0.01:
        print(f"  - {etf_symbols[i]}: {wt:.1%}")


# --- 4e. Hierarchical Risk Parity (HRP) ---
print("\n--- Section Y: Hierarchical Risk Parity Portfolio Optimization ---")

def correl_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

def get_quasi_diag(link):
    sort_ix = leaves_list(link)
    return sort_ix

# Step 1: Compute correlation and distance matrix
corr = returns_monthly.corr()
dist = correl_dist(corr)
link = linkage(squareform(dist), method='single')

# Step 2: Quasi-diagonalization (ordering)
sort_ix = get_quasi_diag(link)
sorted_tickers = [etf_symbols[i] for i in sort_ix]

# Step 3: Inverse variance portfolio within clusters
cov = returns_monthly[sorted_tickers].cov().values * 12
inv_diag = 1 / np.diag(cov)
weights = inv_diag / np.sum(inv_diag)

# Map back to original order
hrp_weights = np.zeros(len(etf_symbols))
for i, idx in enumerate(sort_ix):
    hrp_weights[idx] = weights[i]

print("Optimal HRP Portfolio Weights:")
for i, wt in enumerate(hrp_weights):
    if wt > 0.01:
        print(f"  - {etf_symbols[i]}: {wt:.1%}")

# Compute cumulative returns
hrp_returns = returns_monthly[etf_symbols] @ hrp_weights
cum_hrp = (1 + hrp_returns).cumprod()


# ==============================================================================
# SECTION 5: REGIME-SWITCHING MODEL
# ==============================================================================
print("\n--- Section 5: Building a Market Regime-Switching Model ---")

# The market does not behave uniformly; it switches between different states or "regimes".
# We will build a model to identify these regimes based on the market's own behavior
# (using VOO returns) and external economic indicators.

# --- 5a. Load Exogenous Economic Data ---
# We use VIX (volatility index) and US Treasury yields as indicators of the
# broader economic environment.
start_date = returns_monthly.index.min()
end_date = returns_monthly.index.max()

def get_yield_curve(start, end):
    """Fetches US Treasury yield data from FRED."""
    print("Fetching yield curve data...")
    symbols = {'3M': "DGS3MO", '10Y': "DGS10"}
    df = pd.DataFrame()
    for label, fred_code in symbols.items():
        try:
            data = DataReader(fred_code, 'fred', start, end)
            df[label] = data[fred_code]
        except Exception:
            df[label] = np.nan
    df = df / 100.0  # Convert from percent to decimal
    df['Spread_10Y_3M'] = df['10Y'] - df['3M']
    return df

# Fetch VIX and Yield Curve data.
yield_curve_df = get_yield_curve(start_date, end_date)
vix_df = yf.Ticker('^VIX').history(start=start_date, end=end_date)[['Close']]
vix_df.rename(columns={'Close': 'VIX'}, inplace=True)

# --- 5b. Align and Prepare Data for Modeling ---
# Combine all external data and align it to our monthly return frequency.
exog_df = pd.concat([yield_curve_df, vix_df], axis=1)
exog_df = exog_df.resample('M').last().ffill().dropna()

# Find the common date range between our ETF returns and the economic data.
common_index = returns_monthly.index.intersection(exog_df.index)
returns_aligned = returns_monthly.loc[common_index]
exog_aligned = exog_df.loc[common_index]

# We use *lagged* economic data to predict the *next* month's regime.
# This ensures our model is not using future information.
exog_lagged = exog_aligned.shift(1).dropna()

# Final alignment after lagging.
final_index = returns_aligned.index.intersection(exog_lagged.index)
returns_final = returns_aligned.loc[final_index]
exog_final_lagged = exog_lagged.loc[final_index]

# The model will identify regimes based on the returns of the broad market (VOO).
endog_voo = returns_final['VOO']

print(f"Final dataset for regime modeling has {len(endog_voo)} monthly observations.")

# --- 5c. Fit the Markov Regime-Switching Model ---
# We will test models with different numbers of regimes (e.g., 2, 3, 4) and
# select the best one based on the Bayesian Information Criterion (BIC), which
# balances model fit with model complexity.
models = {}
for k in range(2, MAX_REGIMES_TO_TEST + 1):
    print(f"Fitting model with {k} regimes...")
    try:
        # This model allows both the mean return and the volatility to be different in each regime.
        # The 'exog_tvtp' allows the economic data to influence the probability of switching regimes.
        mod = MarkovRegression(
            endog=endog_voo,
            k_regimes=k,
            trend='c',  # Constant mean term
            switching_variance=True,
            exog_tvtp=sm.add_constant(exog_final_lagged)
        )
        res = mod.fit(search_reps=20) # Search for the best starting parameters

        # --- Validation Check ---
        # Ensure that each identified regime has a sufficient number of observations.
        assigned_regimes = res.smoothed_marginal_probabilities.idxmax(axis=1)
        counts = assigned_regimes.value_counts()
        if (counts < MIN_OBS_PER_REGIME).any():
            print(f"  > Model with {k} regimes rejected: A regime had insufficient observations.")
            continue

        models[k] = res
        print(f"  > Model with {k} regimes is valid. BIC: {res.bic:.2f}")
    except Exception as e:
        print(f"  > Failed to fit model with {k} regimes: {e}")

if not models:
    raise RuntimeError("No suitable regime-switching models could be fitted.")

# Select the model with the lowest BIC.
best_k = min(models, key=lambda k: models[k].bic)
best_model_results = models[best_k]
print(f"\nBest model selected: {best_k} regimes (Lowest BIC = {best_model_results.bic:.2f})")


# --- 5d. Interpret and Label the Regimes ---
# To make the regimes interpretable, we sort them by their volatility (sigma).
# This gives us consistent labels, e.g., "Regime 0" is always the lowest volatility state.
regime_vols = best_model_results.params.filter(like='sigma2').sort_values()
regime_order = regime_vols.index.str.extract(r'\[(\d+)\]')[0].astype(int)
regime_map = {old_idx: new_idx for new_idx, old_idx in enumerate(regime_order)}

# Display the characteristics (mean return and volatility) of each sorted regime.
sorted_params = pd.DataFrame()
for i in range(best_k):
    original_idx = regime_order.iloc[i]
    mean = best_model_results.params[f'const[{original_idx}]'] * 12 * 100 # Annualized %
    vol = np.sqrt(best_model_results.params[f'sigma2[{original_idx}]']) * np.sqrt(12) * 100 # Annualized %
    sorted_params[f'Regime {i}'] = [f'{mean:.1f}%', f'{vol:.1f}%']

sorted_params.index = ['Annualized Mean (VOO)', 'Annualized Volatility (VOO)']
print("\nCharacteristics of Identified Market Regimes (Sorted by Volatility):")
print(sorted_params)

# Get the final, sorted series of regime probabilities and assignments.
smoothed_probs = best_model_results.smoothed_marginal_probabilities.rename(columns=regime_map).sort_index(axis=1)
regime_series = smoothed_probs.idxmax(axis=1).rename('regime')

# ==============================================================================
# TVTP Analysis: Extract and Visualize Time-Varying Transition Probabilities
# ==============================================================================

# --- Transition Probability Tensor ---
# This tensor stores the full set of time-varying transition probabilities estimated by the model.
# Dimensions: (T, k_regimes, k_regimes)
# For each time step t, transition_probs[t, i, j] gives the probability of moving FROM regime i TO regime j.
transition_probs = best_model_results.transition_matrices

# --- Extract Coefficients of the Transition Models ---
# The Markov model uses logistic regressions (logit) to model how predictors affect the probability of switching regimes.
# These logistic regressions are estimated for each transition pair (origin → target), e.g., 0→1, 1→2, etc.

# Select only the parameters that relate to the exogenous variables affecting transition probs
tvtp_params = best_model_results.params.filter(like='exog_tvtp').copy()

# Convert the parameter index (e.g., 'exog_tvtp[const] transition 0->1') into a DataFrame for easier parsing
tvtp_coeffs = tvtp_params.reset_index()

# Extract numerical identifiers for the target and origin regimes from the string index
# Example: From 'transition 0->1' extract origin=0 and target=1
tvtp_coeffs[['target', 'origin']] = tvtp_coeffs['index'].str.extract(r'(\d+)->(\d+)').astype(int)

# Extract the name of the explanatory variable (e.g., 'const', 'vix', 'slope') used in the transition model
tvtp_coeffs['variable'] = tvtp_coeffs['index'].str.extract(r'exog_tvtp\[(.*?)\]')

# Pivot the long table into a matrix format:
# Rows: (origin, target), Columns: predictor variables
# This allows you to see the logistic regression coefficients for each transition
tvtp_coeffs = tvtp_coeffs.pivot_table(index=['origin', 'target'], columns='variable', values=0)

# --- Display ---
# This prints the logistic regression coefficients that govern the transition dynamics.
# Positive values mean the variable increases the probability of that regime transition.
print("\nTransition Coefficients (per transition origin→target):")
print(tvtp_coeffs.round(3))

# ==============================================================================
# Visualization of Time-Varying Transition Probabilities into Each Regime
# ==============================================================================

# Create one subplot per regime, showing the probability of entering that regime over time
fig, axes = plt.subplots(best_k, 1, figsize=(12, 3.5 * best_k), sharex=True)

# Loop over each regime (j), plotting P(→ j) — the probability of transitioning *into* regime j
for j in range(best_k):
    # At each time t, take the average probability of transitioning into regime j from any other regime i
    # transition_probs[t, i, j] is the probability of i→j
    trans_probs_to_j = pd.Series(
        transition_probs[:, :, j].mean(axis=1),  # Average across all i for fixed j
        index=returns_final.index               # Time index
    )

    # Plot the series for this regime
    axes[j].plot(trans_probs_to_j, label=f'Prob(→ Regime {j})', color='crimson')
    axes[j].set_title(f'Time-Varying Transition Probability into Regime {j}')
    axes[j].set_ylabel('Probability')
    axes[j].legend()
    axes[j].grid(True)

# Label the x-axis once (shared across subplots)
plt.xlabel("Time")
plt.tight_layout()
plt.show()


# ==============================================================================
# SECTION 6: REGIME-AWARE DYNAMIC STRATEGY
# ==============================================================================
print("\n--- Section 6: Building and Backtesting the Dynamic Strategy ---")

# --- 6a. Calculate Regime-Specific Optimal Portfolios ---
# Now, we compute a separate efficient frontier and optimal portfolio for each regime.
# The idea is to hold the best possible portfolio for the current market environment.
regime_frontiers = {}
regime_optimal_weights = {}

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, best_k))

for i in range(best_k):
    print(f"\nAnalyzing Regime {i}...")
    in_regime_periods = (regime_series == i)

    # We need enough data points in a regime to get reliable estimates.
    if in_regime_periods.sum() < max(12, len(etf_symbols)):
        print(f"  > Skipping Regime {i}, not enough data points ({in_regime_periods.sum()}).")
        # Fallback to a 100% VOO portfolio for this regime if we can't optimize.
        regime_optimal_weights[i] = np.array([1.0 if s == 'VOO' else 0.0 for s in etf_symbols])
        continue

    # Estimate parameters using only the data from this regime.
    returns_regime = returns_final[in_regime_periods]
    mu_regime = returns_regime.mean().values * 12 - expense_vector
    cov_regime = LedoitWolf().fit(returns_regime.values).covariance_ * 12

    # Generate the efficient frontier for this regime.
    ef_regime = efficient_frontier(cov_regime, mu_regime, n_points=FRONTIER_POINTS)
    regime_frontiers[i] = ef_regime

    # Find the optimal portfolio by matching the overall VOO benchmark's volatility.
    _, w_opt = select_portfolio(ef_regime, 'sigma', voo_sigma_annual)

    if w_opt is not None:
        regime_optimal_weights[i] = w_opt
        print(f"  > Optimal Portfolio for Regime {i} (matching VOO vol):")
        top_indices = np.argsort(w_opt)[-3:][::-1]
        for idx in top_indices:
            if w_opt[idx] > 0.01:
                print(f"    - {etf_symbols[idx]}: {w_opt[idx]:.1%}")

        # Plot the frontier and the selected optimal point.
        plt.plot(ef_regime['sigma'], ef_regime['mu'], label=f'Regime {i} Frontier', color=colors[i], lw=2)
        opt_sigma = np.sqrt(w_opt @ cov_regime @ w_opt)
        opt_mu = w_opt @ mu_regime
        plt.scatter(opt_sigma, opt_mu, marker='*', s=250, color=colors[i], zorder=5, edgecolors='black')
    else:
        print("  > Could not find an optimal portfolio. Using VOO as fallback.")
        regime_optimal_weights[i] = np.array([1.0 if s == 'VOO' else 0.0 for s in etf_symbols])

# Finalize and show the plot of all regime frontiers.
plt.scatter([voo_sigma_annual], [voo_mu_annual], color='black', marker='X', s=200, label='VOO (Overall)', zorder=5)
plt.title('Efficient Frontiers for Each Market Regime')
plt.xlabel('Annualized Volatility (Sigma)')
plt.ylabel('Annualized Return (Mu)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6b. Backtest the Dynamic Strategy ---
# At each month, our portfolio is a blend of the regime-optimal portfolios,
# weighted by the probability of being in each regime at that time.
dynamic_weights = []
for t in range(len(returns_final)):
    # Get the smoothed probabilities for this time step.
    probs_t = smoothed_probs.iloc[t]
    blended_w = np.zeros(len(etf_symbols))
    # Create the blended portfolio.
    for i in range(best_k):
        if i in regime_optimal_weights:
            blended_w += probs_t[i] * regime_optimal_weights[i]
    # Normalize weights to ensure they sum to 1.
    blended_w /= blended_w.sum()
    dynamic_weights.append(blended_w)

# Calculate the monthly returns of this dynamic portfolio.
dynamic_port_returns = np.sum(np.array(dynamic_weights) * returns_final[etf_symbols].values, axis=1)
dynamic_returns_series = pd.Series(dynamic_port_returns, index=returns_final.index)


# ==============================================================================
# SECTION 7: FINAL PERFORMANCE COMPARISON
# ==============================================================================
print("\n--- Section 7: Comparing All Strategies ---")

# --- 7a. Define a Performance Metrics Calculator ---
def calculate_metrics(returns_series):
    """
    Calculates key performance metrics for a series of returns.
    """
    # Total Cumulative Return
    cumulative_returns = (1 + returns_series).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100

    # Annualized Return (Geometric)
    ann_return = ((1 + returns_series.mean()) ** 12 - 1) * 100

    # Annualized Volatility
    ann_vol = returns_series.std() * np.sqrt(12) * 100

    # Sharpe Ratio (assumes risk-free rate is 0)
    # Measures risk-adjusted return. Higher is better.
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan

    # Maximum Drawdown
    # The largest peak-to-trough drop in portfolio value. A measure of downside risk.
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak - 1) * 100
    max_drawdown = drawdown.min()

    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": ann_return,
        "Annualized Volatility (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown
    }

# --- 7b. Prepare All Strategy Returns for Comparison ---
# We need to calculate the historical returns for each static portfolio to compare
# them against the dynamic strategy and the VOO benchmark.
strategies = {
    'VOO Benchmark': returns_final['VOO'],
    'Static Raw (Sigma-Match)': (returns_final[etf_symbols] @ w_sigma_raw) if w_sigma_raw is not None else pd.Series(np.nan, index=returns_final.index),
    'Static Shrunk (Sigma-Match)': (returns_final[etf_symbols] @ w_sigma_shrunk) if w_sigma_shrunk is not None else pd.Series(np.nan, index=returns_final.index),
    'Static Resampled': (returns_final[etf_symbols] @ w_resampled) if w_resampled is not None else pd.Series(np.nan, index=returns_final.index),
    'Black-Litterman': (returns_final[etf_symbols] @ w_bl_opt) if 'w_bl_opt' in globals() else pd.Series(np.nan, index=returns_final.index),
    'Hierarchical Risk Parity': (returns_final[etf_symbols] @ hrp_weights) if 'hrp_weights' in globals() else pd.Series(np.nan, index=returns_final.index),
    'Dynamic Regime Strategy': dynamic_returns_series
}

# --- 7c. Interactive Strategy Comparison Dashboard ---
def plot_strategy_comparison(selected_strategies):
    plt.figure(figsize=(14, 8))
    for name in selected_strategies:
        series = strategies[name].dropna()
        (1 + series).cumprod().plot(label=name, lw=2)

    plt.title('Cumulative Performance Comparison of Selected Strategies', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    display(all_perf_df.loc[selected_strategies])

multi_select = widgets.SelectMultiple(
    options=list(strategies.keys()),
    value=('VOO Benchmark', 'Black-Litterman', 'Hierarchical Risk Parity'),
    description='Strategies:',
    rows=7,
    layout=widgets.Layout(width='50%')
)

interactive_output = widgets.interactive_output(plot_strategy_comparison, {'selected_strategies': multi_select})

print("\nSelect strategies to visualize interactively:")
display(widgets.HBox([multi_select]), interactive_output)

# --- 7d. Plot Cumulative Performance of All Strategies ---
plt.figure(figsize=(14, 8))
for name, returns in strategies.items():
    if not returns.isnull().all():
        (1 + returns).cumprod().plot(label=name, lw=2)

plt.title('Cumulative Performance Comparison of All Strategies', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.yscale('log') # Log scale is useful for comparing long-term growth rates.
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# --- 7e. Display Final Performance Metrics Table ---
all_perf_metrics = {
    name: calculate_metrics(returns.dropna())
    for name, returns in strategies.items()
}
all_perf_df = pd.DataFrame(all_perf_metrics).T

print("\n" + "="*50)
print("      COMPREHENSIVE STRATEGY PERFORMANCE METRICS")
print("="*50)
print(all_perf_df)
print("="*50 + "\n")

# --- 7e. Visualize Regime Probabilities vs. Market Returns ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
smoothed_probs.plot(ax=axes[0], kind='area', stacked=True, colormap='viridis', alpha=0.8)
axes[0].set_title('Smoothed Probabilities of Each Market Regime Over Time', fontsize=14)
axes[0].set_ylabel('Probability')
axes[0].legend(title='Regime', loc='upper left')
axes[0].grid(True, linestyle='--', alpha=0.5)

returns_final['VOO'].plot(ax=axes[1], color='black', label='VOO Monthly Return')
axes[1].set_title('VOO Monthly Returns', fontsize=14)
axes[1].set_ylabel('Return')
axes[1].axhline(0, color='grey', lw=1)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.xlabel("Date")
plt.tight_layout()
plt.show()


