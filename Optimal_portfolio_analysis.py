# ==============================================================================
#
#        PORTFOLIO OPTIMIZATION & DYNAMIC ASSET ALLOCATION
#
# ==============================================================================
#
# OVERVIEW:
#
# This script performs a comprehensive portfolio analysis using various optimization
# techniques. It begins with classic Mean-Variance Optimization and then explores
# more advanced methods to create robust and dynamic asset allocation strategies.
#
# The script is divided into the following main sections:
#
#   1.  SETUP & CONFIGURATION:
#       -   Loads necessary libraries and defines global settings.
#
#   2.  DATA LOADING & PREPARATION:
#       -   Fetches a list of Vanguard ETFs and their metadata via web scraping.
#       -   Downloads historical price data for these ETFs from Yahoo Finance.
#       -   Processes the data to calculate monthly returns over a specified window.
#
#   3.  STATIC PORTFOLIO ANALYSIS (MEAN-VARIANCE OPTIMIZATION):
#       -   Calculates expected returns and covariance, applying "shrinkage"
#           techniques (Ledoit-Wolf) for more stable estimates.
#       -   Computes the "Efficient Frontier" to find optimal portfolios.
#       -   Applies L1 regularization to remove small weights.
#       -   Constructs several static portfolios and compares them to a benchmark (VOO).
#
#   4.  ADVANCED STATIC MODELS & ROBUSTNESS CHECKS:
#       -   Implements Resampled Efficient Frontier, Rolling Window analysis,
#           Black-Litterman, Risk Parity, Hierarchical Risk Parity (HRP), and
#           DCC-GARCH models to create more robust portfolios.
#
#   5.  REGIME-SWITCHING MODEL:
#       -   Loads external economic data (VIX, Treasury yields) to help
#           identify distinct market "regimes" (e.g., low-volatility growth,
#           high-volatility decline).
#       -   Fits a Markov Regime-Switching model to the market data.
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
#       -   Runs a Monte Carlo simulation to forecast future performance.
#
# ==============================================================================

# ==============================================================================
# SECTION 1: SETUP & CONFIGURATION
# ==============================================================================
# --- Standard Library Imports ---
import os
import re
import time
import warnings

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import cvxopt as opt
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from arch.univariate import ConstantMean, GARCH
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from pandas_datareader.data import DataReader
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# --- Global Settings & Constants ---
# Set the working directory.
# NOTE: You may need to change this path to your project's root directory.
DIRECTORY = "."
os.chdir(DIRECTORY)

# Analysis Period
ANALYSIS_YEARS = 15

# Optimization & Simulation Parameters
FRONTIER_POINTS = 50  # Number of points to calculate on the efficient frontier.
MC_SIM_SCENARIOS = 10000  # Number of scenarios for Monte Carlo simulation.
MC_SIM_HORIZON_MONTHS = 120  # 10-year horizon for simulation.
RESAMPLE_ITERATIONS = 100  # Number of bootstrap iterations for resampled frontier.
ROLLING_WINDOW_MONTHS = 60  # 5-year rolling window for dynamic weight analysis.

# Regime Modeling Parameters
MIN_OBS_PER_REGIME = 10  # Minimum data points required to consider a regime valid.
MAX_REGIMES_TO_TEST = 4  # Test models with 2 up to this number of regimes.

# --- Initial Setup ---
# Configure pandas for better display
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Configure the CVXOPT solver to not display progress messages
opt.solvers.options["show_progress"] = False

# Suppress convergence warnings from statsmodels fitting procedures
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# SECTION 2: DATA LOADING & PREPARATION
# ==============================================================================
print("--- Section 2: Loading ETF Data and Calculating Returns ---")


# --- 2a. Load ETF Metadata Directly from Vanguard Website ---
# We use Selenium to scrape the Vanguard advisor site for an up-to-date list of
# tickers, names, and expense ratios.
def extract_etf_data_from_page(driver_instance):
    """
    Extracts ETF data (Symbol, Name, Expense Ratio) from the currently
    viewed table on the Vanguard website.

    Args:
        driver_instance: The active Selenium WebDriver instance.

    Returns:
        A list of dictionaries, where each dictionary represents an ETF.
    """
    data = []
    try:
        table_body = driver_instance.find_element(By.CSS_SELECTOR, "table tbody")
        rows = table_body.find_elements(By.TAG_NAME, "tr")
        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                symbol = cells[0].text.strip()
                # Fund name is in a nested div; clean up extraneous text like "NEW FUND"
                raw_name = row.find_elements(By.TAG_NAME, "div")[1].text.strip()
                fund_name = re.sub(
                    r"\s*(NEW FUND)?\s*$", "", raw_name.replace("\n", " ")
                ).strip()
                # Expense ratio is in the 8th column; clean and convert to float
                expense_text = cells[7].text.strip().replace("%", "").strip()
                expense_ratio = float(expense_text) / 100 if expense_text else None
                data.append(
                    {
                        "Symbol": symbol,
                        "Fund name": fund_name,
                        "Expense ratio": expense_ratio,
                    }
                )
            except (IndexError, ValueError):
                # Skip row if any element is missing or fails to parse
                continue
    except Exception as e:
        print(f"Error extracting table data: {e}")
    return data


try:
    print("Scraping ETF data from Vanguard website...")
    # Setup Selenium Chrome driver in headless mode (no visible browser window)
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")

    # IMPORTANT: Update this path to your local chromedriver executable.
    # For better portability, consider using webdriver-manager:
    # from webdriver_manager.chrome import ChromeDriverManager
    # service = Service(ChromeDriverManager().install())
    chrome_service = Service(executable_path="/Users/dominikjurek/Library/CloudStorage/Dropbox/Personal/Investment/chromedriver")
    
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    # Step 1: Load the Vanguard ETF page and wait for it to render
    driver.get("https://advisors.vanguard.com/investments/etfs")
    time.sleep(6)  # Allow time for JavaScript to load the table

    # Step 2: Extract data from the first page
    all_etf_data = extract_etf_data_from_page(driver)

    # Step 3: Click the "Next page" button to load the remaining ETFs
    try:
        next_button = driver.find_element(
            By.XPATH, '//button[@aria-label[contains(., "Forward one page")]]'
        )
        next_button.click()
        time.sleep(5)  # Wait for the second page to load
        all_etf_data += extract_etf_data_from_page(driver)
    except Exception as e:
        print(f"Could not navigate to the second page (or it doesn't exist): {e}")

    # Step 4: Process the scraped data
    df_etf_metadata = pd.DataFrame(all_etf_data)
    etf_name_map = dict(zip(df_etf_metadata["Symbol"], df_etf_metadata["Fund name"]))
    etf_expense_map = dict(
        zip(df_etf_metadata["Symbol"], df_etf_metadata["Expense ratio"])
    )
    etf_symbols = list(etf_name_map.keys())
    print(f"Successfully extracted metadata for {len(etf_symbols)} ETFs.")
    print("Sample of extracted ETF data:")
    print(df_etf_metadata.head())
except Exception as e:
    # If scraping fails, fall back to a predefined list of core ETFs
    print(f"Could not complete web scraping. Reason: {e}")
    print("Falling back to a predefined list of core ETFs.")
    etf_symbols = [
        "VOO", "VTI", "VEA", "VWO", "BND", "BNDX", "VGIT", "VGLT", "VTIP", "MUB",
    ]
    etf_name_map = {s: s for s in etf_symbols}
    etf_expense_map = {s: 0.0003 for s in etf_symbols}  # Use a reasonable default
finally:
    if "driver" in locals() and driver:
        driver.quit()


# --- 2b. Optional: Filter the ETF Universe ---
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


# --- 2c. Fetch Historical Price Data ---
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


# --- 2d. Process and Clean Return Data ---
# Standardize the index to datetime objects without timezone information
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)

# Resample daily prices to month-end, then calculate monthly percentage returns
returns_monthly = all_prices.resample("M").last().pct_change()

# Limit data to the last N years for a more relevant analysis window
cutoff_date = returns_monthly.index.max() - pd.DateOffset(years=ANALYSIS_YEARS)
returns_monthly = returns_monthly[returns_monthly.index >= cutoff_date]

# Data Cleaning:
# 1. Drop ETFs that do not have at least 10 years of non-NA observations
MIN_OBSERVATIONS = 10 * 12
returns_monthly = returns_monthly.dropna(axis=1, thresh=MIN_OBSERVATIONS)

# 2. Drop any month (row) that still has missing values
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

# ==============================================================================
# SECTION 3: STATIC PORTFOLIO ANALYSIS (MEAN-VARIANCE OPTIMIZATION)
# ==============================================================================
print("\n--- Section 3: Performing Static Mean-Variance Optimization ---")


# --- 3a. Estimate Expected Returns and Covariance ---
# These are the two key inputs for Markowitz portfolio optimization.
# 1. Expected Returns (mu): The anticipated annualized return for each asset.
# 2. Covariance Matrix (Sigma): A measure of how asset returns move together.

# Calculate historical annualized mean returns, net of expense ratios
annual_mu_sample = (returns_monthly.mean().values * 12) - expense_vector

# The sample covariance matrix is calculated from historical returns and annualized
sample_cov = returns_monthly.cov().values
annual_cov_sample = sample_cov * 12


# --- 3b. Apply Shrinkage to Improve Covariance Estimates ---
# Shrinkage techniques adjust historical sample estimates to be more stable.
# Ledoit-Wolf shrinkage computes an optimal blend of the sample covariance
# matrix and a more structured, less noisy target matrix.
lw = LedoitWolf().fit(returns_monthly.values)
annual_cov_shrunk = lw.covariance_ * 12

# We will use the sample returns and the shrunk covariance for our primary models
annual_mu = annual_mu_sample
print("Annualized Expected Returns (Sample):")
print(pd.Series(annual_mu, index=etf_symbols).round(4))


# --- 3c. Define the Efficient Frontier Optimizer with L1 Regularization ---
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

    # --- L1 REFORMULATION SETUP ---
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
    ).T

    # --- END OF L1 REFORMULATION SETUP ---

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


# --- 3d. Define Benchmark and Helper Functions ---
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


# ------------------------------------------------------------------------------
#  Grid-Search for the "Best" L1-Penalty (lambda_1)
# ------------------------------------------------------------------------------
# - A simple train/validation split is used (last 20% = validation).
# - The metric for "best" is the out-of-sample Sharpe Ratio.
# - The portfolio on the frontier is chosen by matching VOO's volatility.
# ------------------------------------------------------------------------------
print("\n--- Grid-searching for optimal L1 penalty (lambda) ---")
lambda_grid = np.logspace(-6, -1, 11)  # 11 points: 1e-6 to 1e-1
val_frac = 0.20
target_sig = voo_sigma_annual

# Split data into training and validation sets
T = len(returns_monthly)
split_idx = int((1 - val_frac) * T)
ret_train = returns_monthly.iloc[:split_idx]
ret_val = returns_monthly.iloc[split_idx:]

mu_train = ret_train.mean().values * 12
cov_train = ret_train.cov().values * 12
mu_val = ret_val.mean().values * 12
cov_val = ret_val.cov().values * 12


def sharpe_ratio(mu, sigma):
    """
    Calculates annualized Sharpe ratio from annualized decimal returns.
    (Assumes risk-free rate = 0).
    """
    return mu / sigma if sigma > 0 else np.nan


def eval_lambda(lam):
    """Fit frontier with lambda_1=lam on train-set, score on validation."""
    front = efficient_frontier(cov_train, mu_train, n_points=30, lambda_l1=lam)
    _, w = select_portfolio(front, "sigma", target_sig)
    if w is None:
        return np.nan
    mu_out_of_sample = w @ mu_val
    sigma_out_of_sample = np.sqrt(w @ cov_val @ w)
    return sharpe_ratio(mu_out_of_sample, sigma_out_of_sample)


# Perform grid search
scores = [eval_lambda(lam) for lam in lambda_grid]
best_idx = int(np.nanargmax(scores))
best_lambda = float(lambda_grid[best_idx])

# Report results
print(f"lambda_1 candidates: {[f'{l:.5g}' for l in lambda_grid]}")
print(f"Validation Sharpe: {[f'{s:.3f}' for s in scores]}")
print(f"\n-> Selected optimal lambda_1 = {best_lambda:.5g}\n")


# --- 3e. Generate Frontiers and Select Key Portfolios ---
print("Generating efficient frontiers...")
# Frontier using simple sample estimates
ef_raw = prune_frontier(
    efficient_frontier(annual_cov_sample, annual_mu, n_points=FRONTIER_POINTS)
)
# Frontier using L1 regularization
ef_reg_l1 = prune_frontier(
    efficient_frontier(
        annual_cov_shrunk, annual_mu, n_points=FRONTIER_POINTS, lambda_l1=best_lambda
    )
)
# Frontier using shrinkage-adjusted covariance
ef_shrunk = prune_frontier(
    efficient_frontier(annual_cov_shrunk, annual_mu, n_points=FRONTIER_POINTS)
)

# Find portfolios on each frontier matching the VOO benchmark's risk or return
_, w_mu_raw = select_portfolio(ef_raw, "mu", voo_mu_annual)
_, w_sigma_raw = select_portfolio(ef_raw, "sigma", voo_sigma_annual)
_, w_mu_reg_l1 = select_portfolio(ef_reg_l1, "mu", voo_mu_annual)
_, w_sigma_reg_l1 = select_portfolio(ef_reg_l1, "sigma", voo_sigma_annual)
_, w_mu_shrunk = select_portfolio(ef_shrunk, "mu", voo_mu_annual)
_, w_sigma_shrunk = select_portfolio(ef_shrunk, "sigma", voo_sigma_annual)

# Display the composition of selected portfolios
portfolios_to_display = {
    "Raw (Return-Matched)": w_mu_raw,
    "Raw (Risk-Matched)": w_sigma_raw,
    "L1 Regularized (Return-Matched)": w_mu_reg_l1,
    "L1 Regularized (Risk-Matched)": w_sigma_reg_l1,
    "Shrunk (Return-Matched)": w_mu_shrunk,
    "Shrunk (Risk-Matched)": w_sigma_shrunk,
}

for label, weights in portfolios_to_display.items():
    if weights is not None:
        print(f"\nTop 3 ETFs for {label} Portfolio:")
        top_indices = np.argsort(weights)[-3:][::-1]
        for i in top_indices:
            if weights[i] > 0.001:  # Only show assets with meaningful weight
                symbol = etf_symbols[i]
                name = etf_name_map.get(symbol, "Unknown")
                print(f"  {symbol} ({name}): {weights[i]:.2%}")


# --- 3f. Plot Static Efficient Frontiers ---
plt.figure(figsize=(12, 8))
plt.plot(ef_raw["sigma"], ef_raw["mu"], "o-", label="Raw Estimate Frontier", alpha=0.7)
plt.plot(ef_reg_l1["sigma"], ef_reg_l1["mu"], "o-", label="L1 Regularized Frontier", lw=2)
plt.plot(
    ef_shrunk["sigma"], ef_shrunk["mu"], "o-", label="Shrunk Covariance Frontier", lw=2
)
plt.scatter(
    [voo_sigma_annual],
    [voo_mu_annual],
    color="red",
    marker="X",
    s=200,
    label="VOO Benchmark",
    zorder=5,
)
plt.title(
    "Efficient Frontiers: Raw vs. L1 Regularization vs. Shrunk Covariance", fontsize=16
)
plt.xlabel("Annualized Volatility (sigma)", fontsize=12)
plt.ylabel("Annualized Expected Return (mu)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.show()


# ==============================================================================
# SECTION 4: ADVANCED STATIC MODELS & ROBUSTNESS CHECKS
# ==============================================================================
print("\n--- Section 4: Advanced Static Models & Robustness Checks ---")


# --- 4a. Resampled Efficient Frontier (Bootstrapping) ---
# This technique addresses "estimation error" by creating many new return
# datasets via bootstrapping. The final portfolio is the average of all optimal
# portfolios found, leading to a more diversified and stable allocation.
print(f"Running Resampled Frontier with {RESAMPLE_ITERATIONS} iterations...")
n_obs, n_assets = returns_monthly.shape
resampled_weights_list = []

for i in range(RESAMPLE_ITERATIONS):
    if (i + 1) % 25 == 0:
        print(f"  Resample iteration {i + 1}/{RESAMPLE_ITERATIONS}...")

    # Create a bootstrap sample of the monthly returns
    boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
    returns_boot = returns_monthly.iloc[boot_indices]

    # Recalculate parameters for the bootstrap sample
    mu_boot = (returns_boot.mean().values * 12) - expense_vector
    try:
        # Use shrunk covariance for better stability in each resampled data set
        cov_boot = LedoitWolf().fit(returns_boot.values).covariance_ * 12
        # Generate frontier and select the risk-matched portfolio
        ef_boot = efficient_frontier(cov_boot, mu_boot, n_points=30)
        _, w_boot = select_portfolio(ef_boot, "sigma", voo_sigma_annual)
        if w_boot is not None:
            resampled_weights_list.append(w_boot)
    except (ValueError, np.linalg.LinAlgError):
        # Skip iteration if the solver or covariance estimation fails
        continue

# The final resampled portfolio is the average of weights from all iterations
if resampled_weights_list:
    w_resampled = np.mean(resampled_weights_list, axis=0)
    print("\nTop 3 ETFs for Resampled Portfolio:")
    top_indices = np.argsort(w_resampled)[-3:][::-1]
    for i in top_indices:
        symbol = etf_symbols[i]
        name = etf_name_map.get(symbol, "Unknown")
        print(f"  {symbol} ({name}): {w_resampled[i]:.2%}")
else:
    w_resampled = None
    print("\nCould not generate a resampled portfolio.")


# --- 4b. Rolling Window Estimation ---
# This analysis shows how the optimal allocation would have changed over time,
# providing insight into the strategy's stability.
print(f"\nPerforming Rolling Window analysis with a {ROLLING_WINDOW_MONTHS}-month window...")
rolling_dates = returns_monthly.index[ROLLING_WINDOW_MONTHS:]
rolling_weights_list = []

for date in rolling_dates:
    # Create a data window of the last N months
    window_data = returns_monthly.loc[:date].iloc[-ROLLING_WINDOW_MONTHS:]
    # Estimate parameters on the window
    mu_roll = (window_data.mean().values * 12) - expense_vector
    cov_roll = LedoitWolf().fit(window_data.values).covariance_ * 12
    try:
        # Find the optimal (risk-matched) portfolio for this period
        ef_roll = efficient_frontier(cov_roll, mu_roll, n_points=30)
        _, w_roll = select_portfolio(ef_roll, "sigma", voo_sigma_annual)
        if w_roll is not None:
            rolling_weights_list.append(pd.Series(w_roll, index=etf_symbols, name=date))
    except (ValueError, np.linalg.LinAlgError):
        continue

# Combine results and plot the weight changes for the most important assets
if rolling_weights_list:
    rolling_weights_df = pd.concat(rolling_weights_list, axis=1).T
    # Identify the top 3 ETFs by average weight over time
    top_etfs = rolling_weights_df.mean().sort_values(ascending=False).head(3).index
    rolling_weights_df[top_etfs].plot(
        figsize=(12, 7), title="Top 3 ETF Weights Over Time (Rolling Optimization)", lw=2
    )
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    plt.grid(True, linestyle="--")
    plt.legend(title="ETFs")
    plt.tight_layout()
    plt.show()


# --- 4c. Black-Litterman Optimization ---
# The Black-Litterman model starts with market-implied equilibrium returns and
# then tilts them based on an investor's specific views, creating a blended,
# more intuitive set of expected returns for optimization.
def run_black_litterman(
    shrink_factor=0.3, tau_prior=0.20, omega_scale=0.02, print_summary=True
):
    """
    Performs Black-Litterman optimization with tunable hyperparameters.

    Args:
        shrink_factor (float): Weight on the sample mean in the prior;
                               1-shrink_factor goes to the grand mean.
        tau_prior (float): Prior uncertainty scale. Larger => prior is less certain.
        omega_scale (float): Diagonal element for Omega matrix (view uncertainty).
                             Larger => views are softer/less binding.
        print_summary (bool): If True, prints portfolio weights and top holdings.

    Returns:
        tuple: (weights_bl, mu_posterior)
               - weights_bl (np.ndarray | None): Optimal long-only weights.
               - mu_posterior (np.ndarray | None): Posterior expected-return vector.
    """
    n_assets = len(etf_symbols)
    # 1. Prior mean (pi) - a blend of sample mean and grand mean
    grand_mean = annual_mu_sample.mean()
    prior_mean = shrink_factor * annual_mu_sample + (1 - shrink_factor) * grand_mean

    try:
        # 2. Investor view: each ETF's return is approximately equal to VOO's return
        idx_voo = etf_symbols.index("VOO")
        other_symbols = [sym for sym in etf_symbols if sym != "VOO"]
        # P matrix selects assets involved in the view
        P = np.zeros((len(other_symbols), n_assets))
        for k, sym in enumerate(other_symbols):
            P[k, returns_monthly.columns.get_loc(sym)] = 1  # +1 on ETF
            P[k, idx_voo] = -1  # -1 on VOO
        # Q vector contains the expected outperformance ("approximately 0" in this case)
        Q = np.zeros(len(other_symbols))

        # 3. View-uncertainty (Omega)
        Omega = np.diag(np.full(len(Q), omega_scale))

        # 4. Posterior mean calculation (core Black-Litterman formula)
        inv_cov_prior = np.linalg.inv(tau_prior * annual_cov_shrunk)
        inv_Omega = np.linalg.inv(Omega)
        middle = np.linalg.inv(inv_cov_prior + P.T @ inv_Omega @ P)
        mu_posterior = middle @ (inv_cov_prior @ prior_mean + P.T @ inv_Omega @ Q)

        # 5. Optimize portfolio (mean-variance) using the posterior returns
        P_qp = opt.matrix(annual_cov_shrunk)
        q_qp = opt.matrix(-mu_posterior)
        G_qp = opt.matrix(-np.eye(n_assets))  # w >= 0
        h_qp = opt.matrix(np.zeros(n_assets))
        A_qp = opt.matrix(np.ones((1, n_assets)))  # sum(w) = 1
        b_qp = opt.matrix(1.0)

        sol = opt.solvers.qp(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp)
        if sol["status"] != "optimal":
            raise RuntimeError(f"CVXOPT failed: {sol['status']}")

        weights_bl = np.array(sol["x"]).ravel()
        weights_bl[weights_bl < 1e-7] = 0
        weights_bl /= weights_bl.sum()

        if print_summary:
            print(
                f"\n--- Black-Litterman (shrink={shrink_factor}, "
                f"tau={tau_prior}, omega={omega_scale}) ---"
            )
            print("Weights > 1%:")
            for i, w in enumerate(weights_bl):
                if w > 0.01:
                    print(f"  {etf_symbols[i]}: {w:.1%}")
            top_idx = weights_bl.argsort()[-3:][::-1]
            print("\nTop 3 ETFs:")
            for i in top_idx:
                sym = etf_symbols[i]
                name = etf_name_map.get(sym, "Unknown")
                print(f"  {sym} ({name}): {weights_bl[i]:.2%}")

        return weights_bl, mu_posterior

    except (ValueError, IndexError, np.linalg.LinAlgError) as e:
        if print_summary:
            print(f"Black-Litterman failed: could not locate VOO or matrix error. {e}")
        return None, None


# ------------------------------------------------------------------------------
#  Hyperparameter Grid Search for Black-Litterman
# ------------------------------------------------------------------------------
print("\n--- Tuning Black-Litterman Hyperparameters ---")
shrink_grid = [0.2, 0.4, 0.6, 0.8]
tau_grid = [0.05, 0.10, 0.20, 0.30, 0.40]
omega_grid = [0.005, 0.01, 0.02, 0.05]
records = []

for shrink in shrink_grid:
    for tau in tau_grid:
        for omega in omega_grid:
            w, mu_post = run_black_litterman(shrink, tau, omega, print_summary=False)
            if w is None:
                continue
            exp_ret = float(w @ mu_post)
            vol = float(np.sqrt(w @ annual_cov_shrunk @ w))
            sharpe = exp_ret / vol if vol else np.nan
            records.append(
                {
                    "shrink": shrink,
                    "tau": tau,
                    "omega": omega,
                    "exp_return": exp_ret,
                    "vol": vol,
                    "sharpe": sharpe,
                    "top_weight": w.max(),
                    "assets_>1pct": (w > 0.01).sum(),
                }
            )

tune_df = pd.DataFrame(records).sort_values("sharpe", ascending=False)
print("Tuning results (top 10 by in-sample Sharpe):")
print(tune_df.head(10))
# ------------------------------------------------------------------------------

# Rerun with the best parameters found and print a summary
print("\n--- Final Black-Litterman Portfolio (using best tuned parameters) ---")
best_row = tune_df.iloc[0]
w_bl_opt, mu_bl = run_black_litterman(
    best_row["shrink"], best_row["tau"], best_row["omega"], print_summary=True
)


# --- 4d. Risk Parity Optimization ---
# Risk Parity aims to construct a portfolio where each asset contributes equally
# to the total portfolio risk. It ignores expected returns.
print("\n--- Risk Parity Portfolio Optimization ---")


def portfolio_volatility(weights, cov_matrix):
    """Calculates the annualized volatility of a portfolio."""
    return np.sqrt(weights.T @ cov_matrix @ weights)


def risk_contributions(weights, cov_matrix):
    """Calculates each asset's percentage contribution to total portfolio risk."""
    port_vol = portfolio_volatility(weights, cov_matrix)
    if port_vol == 0:
        return np.zeros_like(weights)
    # Marginal Risk Contribution (MRC) = (Cov * w) / sigma_p
    mrc = (cov_matrix @ weights) / port_vol
    # Total Risk Contribution = w_i * MRC_i
    return weights * mrc


def risk_parity_objective(weights, cov_matrix):
    """
    Objective function for the optimizer. It seeks to minimize the
    variance of risk contributions across all assets, forcing them to be equal.
    """
    total_risk_contribs = risk_contributions(weights, cov_matrix)
    # Target is an equal contribution from each asset
    target_contribution = total_risk_contribs.sum() / len(weights)
    # Minimize the squared differences from this target
    return np.sum((total_risk_contribs - target_contribution) ** 2)


# Solve the optimization problem
initial_weights = np.ones(n_assets) / n_assets
bounds = tuple((0.0, 1.0) for _ in range(n_assets))
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

result = minimize(
    fun=risk_parity_objective,
    x0=initial_weights,
    args=(annual_cov_sample,),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"disp": False},
)

rp_weights = result.x if result.success else None
if rp_weights is not None:
    print("Optimal Risk Parity Portfolio (weights > 1%):")
    for i, weight in enumerate(rp_weights):
        if weight > 0.01:
            print(f"  - {etf_symbols[i]}: {weight:.1%}")
    print("\nTop 3 ETFs for Risk Parity Portfolio:")
    top_indices = np.argsort(rp_weights)[-3:][::-1]
    for i in top_indices:
        symbol = etf_symbols[i]
        name = etf_name_map.get(symbol, "Unknown")
        print(f"  {symbol} ({name}): {rp_weights[i]:.2%}")
else:
    print("Risk Parity optimization failed.")


# --- 4e. Hierarchical Risk Parity (HRP) Optimization ---
# HRP uses graph theory and machine learning to build a diversified portfolio.
# It clusters assets, reorders the covariance matrix, and allocates weights recursively.
print("\n--- Hierarchical Risk Parity Portfolio Optimization ---")


def correlation_to_distance(corr_matrix):
    """Converts a correlation matrix to a distance matrix."""
    return np.sqrt(0.5 * (1 - corr_matrix))


def get_cluster_variance(cov_matrix, cluster_indices):
    """Calculates the variance of a cluster of assets."""
    sub_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
    # Inverse variance weights within the cluster
    inv_var_weights = 1.0 / np.diag(sub_cov)
    inv_var_weights /= inv_var_weights.sum()
    return inv_var_weights.T @ sub_cov @ inv_var_weights


def recursive_bisection(cov_matrix, sorted_indices):
    """Recursively splits weights between asset clusters."""
    weights = pd.Series(1.0, index=sorted_indices)
    clusters = [sorted_indices]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        # Bisect the cluster
        split_point = len(cluster) // 2
        left_cluster, right_cluster = cluster[:split_point], cluster[split_point:]
        # Calculate variance for each sub-cluster
        var_left = get_cluster_variance(cov_matrix, left_cluster)
        var_right = get_cluster_variance(cov_matrix, right_cluster)
        # Allocate weights inversely to cluster variance
        alpha = 1.0 - var_left / (var_left + var_right)
        weights[left_cluster] *= alpha
        weights[right_cluster] *= 1.0 - alpha
        # Add the new sub-clusters to the list to be processed
        clusters.extend([left_cluster, right_cluster])
    return weights.sort_index()


# Step 1: Hierarchical Clustering
corr_matrix = returns_monthly.corr()
dist_matrix = correlation_to_distance(corr_matrix)
linkage_matrix = linkage(squareform(dist_matrix), method="single")

# Step 2: Quasi-Diagonalization (Seriation)
# This reorders assets to place similar assets next to each other.
sorted_indices = leaves_list(linkage_matrix)
sorted_tickers = [etf_symbols[i] for i in sorted_indices]

# Step 3: Recursive Bisection
sorted_cov = returns_monthly[sorted_tickers].cov().values * 12
hrp_weights_sorted = recursive_bisection(sorted_cov, np.arange(len(sorted_tickers)))

# Map weights back to the original order of etf_symbols
hrp_weights = np.zeros(n_assets)
for i, ticker in enumerate(sorted_tickers):
    original_idx = etf_symbols.index(ticker)
    hrp_weights[original_idx] = hrp_weights_sorted.iloc[i]

print("Optimal HRP Portfolio (weights > 1%):")
for i, weight in enumerate(hrp_weights):
    if weight > 0.01:
        print(f"  - {etf_symbols[i]}: {weight:.1%}")

if hrp_weights is not None:
    print("\nTop 3 ETFs for HRP Portfolio:")
    top_indices = np.argsort(hrp_weights)[-3:][::-1]
    for i in top_indices:
        symbol = etf_symbols[i]
        name = etf_name_map.get(symbol, "Unknown")
        print(f"  {symbol} ({name}): {hrp_weights[i]:.2%}")

# Optional: Visualize the asset hierarchy with a dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=sorted_tickers, leaf_rotation=90)
plt.title("HRP Asset Clustering (Dendrogram)", fontsize=16)
plt.tight_layout()
plt.show()

# Blend the HRP portfolio with the MVO portfolio (risk-matched to VOO)
# to create a new portfolio that meets VOO's expected return.
mu_hrp = hrp_weights @ annual_mu_sample
mu_mv = w_sigma_raw @ annual_mu_sample
# Calculate blending factor 'alpha'
alpha = np.clip((voo_mu_annual - mu_hrp) / (mu_mv - mu_hrp + 1e-12), 0, 1)
# Create blended weights
w_tilt = (1 - alpha) * hrp_weights + alpha * w_sigma_raw
w_tilt /= w_tilt.sum()
sigma_tilt = np.sqrt(w_tilt @ annual_cov_sample @ w_tilt)

print(f"\nBlended HRP-MVO portfolio: alpha = {alpha:.3f}")
print(f"  Expected mu   = {w_tilt @ annual_mu:.2%} (VOO: {voo_mu_annual:.2%})")
print(f"  Volatility sigma = {sigma_tilt:.2%} (VOO: {voo_sigma_annual:.2%})")
print("\nTop 3 ETFs for Blended HRP-MVO Portfolio:")
top_indices = np.argsort(w_tilt)[-3:][::-1]
for i in top_indices:
    symbol = etf_symbols[i]
    name = etf_name_map.get(symbol, "Unknown")
    print(f"  {symbol} ({name}): {w_tilt[i]:.2%}")


# --- 4f. DCC-GARCH Portfolio Optimization ---
# ------------------------------------------------------------------------------
#  DCC-GARCH Helpers
# ------------------------------------------------------------------------------
# Dynamic Conditional Correlation (DCC) GARCH models capture time-varying
# volatility and correlation, which is useful as correlations often spike
# during market crises.
#
# NOTE: The following is a custom implementation of a DCC(1,1) model.
# I do not know if this implementation is perfectly correct from a financial
# econometrics standpoint. Its correctness should be verified for production use.
# ------------------------------------------------------------------------------
def _dcc_negloglik(params, std_resids):
    """Negative log-likelihood of a DCC(1,1) model given standardized residuals."""
    T, N = std_resids.shape
    alpha, beta = params
    Qbar = np.cov(std_resids.T)  # Unconditional correlation
    Qt = Qbar.copy()  # Initialize Qt_0
    loglik = 0.0
    for t in range(T):
        eps_t = std_resids[t][:, None]  # (N,1)
        Qt = (1 - alpha - beta) * Qbar + alpha * (eps_t @ eps_t.T) + beta * Qt
        diag_sqrt = np.sqrt(np.diag(Qt))
        diag_sqrt[diag_sqrt <= 1e-12] = 1e-12
        Rt = Qt / np.outer(diag_sqrt, diag_sqrt)
        inv_Rt = np.linalg.inv(Rt)
        logdet_Rt = np.log(np.linalg.det(Rt))
        loglik += logdet_Rt + eps_t.T @ inv_Rt @ eps_t
    return 0.5 * loglik.item()


def fit_dcc(std_resids, bounds=((1e-6, 1 - 1e-6),) * 2):
    """Estimates (alpha, beta) by QML; returns params and last Qt, Rt."""
    best_val, best_ab = np.inf, None
    grid = np.linspace(0.01, 0.15, 5)
    for a in grid:
        for b in grid:
            if a + b < 0.999:
                val = _dcc_negloglik((a, b), std_resids)
                if val < best_val:
                    best_val, best_ab = val, (a, b)
    res = minimize(
        _dcc_negloglik,
        x0=np.array(best_ab),
        args=(std_resids,),
        bounds=bounds,
        constraints={"type": "ineq", "fun": lambda x: 0.999 - (x[0] + x[1])},
    )
    alpha, beta = max(res.x[0], 0), max(res.x[1], 0)
    if alpha + beta >= 0.999:
        beta = 0.999 - alpha - 1e-6

    Qbar = np.cov(std_resids.T)
    Qt = Qbar.copy()
    for eps in std_resids:
        eps = eps[:, None]
        Qt = (1 - alpha - beta) * Qbar + alpha * (eps @ eps.T) + beta * Qt
    Rt = Qt / np.outer(np.sqrt(np.diag(Qt)), np.sqrt(np.diag(Qt)))
    return alpha, beta, Qt, Rt


def forecast_dcc_cov(std_resids, cond_vars, horizon=1):
    """Produces a (horizon-step) ahead annualized covariance forecast."""
    alpha, beta, Qt_last, _ = fit_dcc(std_resids)
    Qbar = np.cov(std_resids.T)
    Qt_h = Qt_last.copy()
    for _ in range(horizon):
        Qt_h = (1 - alpha - beta) * Qbar + beta * Qt_h
    Rt_h = Qt_h / np.outer(np.sqrt(np.diag(Qt_h)), np.sqrt(np.diag(Qt_h)))
    var_h = cond_vars[-1]
    cov_h = np.outer(np.sqrt(var_h), np.sqrt(var_h)) * Rt_h
    return cov_h


print("\n--- Rolling DCC-GARCH Optimization ---")
try:
    # Step 1: Prepare weekly data for better GARCH model estimation
    surviving_etf_symbols = returns_monthly.columns.tolist()
    all_prices_daily = pd.concat(
        [get_total_return_series(t) for t in surviving_etf_symbols], axis=1
    ).tz_localize(None)
    returns_weekly = (
        all_prices_daily[surviving_etf_symbols]
        .resample("W-FRI")
        .last()
        .pct_change()
        .loc[returns_monthly.index.min() :]
        .dropna(how="all")
    )

    print("\n--- Testing for GARCH(1,1) Effects in Weekly Returns ---")
    for symbol in surviving_etf_symbols:
        series = returns_weekly[symbol].dropna()
        if len(series) < 20: continue
        lm_test = het_arch(series, nlags=12)
        print(f"{symbol}: ARCH LM Test p-value = {lm_test[1]:.4f}", end="")
        print("  -> ARCH effects detected" if lm_test[1] < 0.05 else "  -> No significant ARCH effects")

    # Step 2: Perform rolling DCC optimization
    month_ends = returns_weekly.resample("M").last().index
    voo_sigma_annual_weekly = returns_weekly["VOO"].std() * np.sqrt(52)
    rolling_dcc_weights_list = []

    for date in month_ends:
        data_window = returns_weekly.loc[:date]
        if len(data_window) < 104: continue  # Need enough data

        try:
            print(f"Optimizing DCC portfolio for month-end {date.date()}...")
            std_resids, cond_vars = [], []
            scale = 100.0  # Rescale returns for better model fitting
            for s in etf_symbols:
                series = scale * data_window[s].dropna()
                am = ConstantMean(series)
                am.volatility = GARCH(1, 1)
                res = am.fit(disp="off")
                std_resids.append(res.std_resid)
                cond_vars.append((res.conditional_volatility / scale) ** 2)

            std_resids = np.column_stack(std_resids)
            cond_vars = np.column_stack(cond_vars)

            forecasted_cov = forecast_dcc_cov(std_resids, cond_vars) * 52
            forecasted_mu = data_window.mean().values * 52

            ef_dcc = efficient_frontier(forecasted_cov, forecasted_mu, n_points=30)
            _, w_dcc = select_portfolio(ef_dcc, "sigma", voo_sigma_annual_weekly)
            if w_dcc is not None:
                rolling_dcc_weights_list.append(pd.Series(w_dcc, index=etf_symbols, name=date))
        except Exception as e:
            print(f"  Skipped {date.date()} due to error: {e}")

    # Step 3: Process results
    if rolling_dcc_weights_list:
        dcc_weights_df = pd.concat(rolling_dcc_weights_list, axis=1).T
        top_dcc_etfs = dcc_weights_df.mean().sort_values(ascending=False).head(3).index
        dcc_weights_df[top_dcc_etfs].plot(
            figsize=(12, 7),
            title="Top 3 ETF Weights Over Time (DCC-GARCH Optimization)",
            lw=2,
        )
        plt.show()

        # Create monthly return series for the DCC strategy
        dcc_shifted_weights = dcc_weights_df.shift(1).reindex(returns_monthly.index).ffill()
        common_idx = dcc_shifted_weights.dropna().index.intersection(returns_monthly.index)
        aligned_weights = dcc_shifted_weights.loc[common_idx]
        aligned_returns = returns_monthly.loc[common_idx]
        dynamic_returns_series_dcc = pd.Series(
            np.sum(aligned_weights.values * aligned_returns[etf_symbols].values, axis=1),
            index=common_idx,
        )
    else:
        dynamic_returns_series_dcc = pd.Series(dtype=float)
        print("No successful DCC-GARCH optimizations were completed.")

except Exception as e:
    print(f"Failed to run DCC-GARCH analysis. Error: {e}")
    dynamic_returns_series_dcc = pd.Series(dtype=float)


# ==============================================================================
# SECTION 5: REGIME-SWITCHING MODEL
# ==============================================================================
print("\n--- Section 5: Building a Market Regime-Switching Model ---")
# --- 5a. Load Exogenous Economic Data ---
# We use VIX and the Treasury yield spread as indicators of the economic environment.
def get_fred_data(start, end):
    """Fetches US Treasury yield and VIX data from FRED."""
    print("Fetching VIX and yield curve data from FRED...")
    symbols = {"3M": "DGS3MO", "10Y": "DGS10", "VIX": "VIXCLS"}
    try:
        df = DataReader(list(symbols.values()), "fred", start, end)
        df = df.rename(columns={v: k for k, v in symbols.items()})
        df[["3M", "10Y"]] = df[["3M", "10Y"]] / 100.0  # Convert to decimal
        df["Spread_10Y_3M"] = df["10Y"] - df["3M"]
        return df.dropna()
    except Exception as e:
        print(f"Could not fetch FRED data: {e}")
        return pd.DataFrame()


exog_df = get_fred_data(returns_monthly.index.min(), returns_monthly.index.max())

# --- 5b. Align and Prepare Data for Modeling ---
# Align all data to our monthly return frequency
exog_monthly = exog_df.resample("M").last().ffill()
common_index = returns_monthly.index.intersection(exog_monthly.index)
returns_aligned = returns_monthly.loc[common_index]
exog_aligned = exog_monthly.loc[common_index]

# Use LAGGED economic data to predict the NEXT month's regime to prevent lookahead bias
exog_lagged = exog_aligned.shift(1).dropna()
final_index = returns_aligned.index.intersection(exog_lagged.index)
returns_final = returns_aligned.loc[final_index]
exog_final_lagged = exog_lagged.loc[final_index, ["VIX", "Spread_10Y_3M"]]
endog_voo = returns_final["VOO"]  # Market returns (VOO) drive regime identification

print(f"Final dataset for regime modeling has {len(endog_voo)} monthly observations.")

# --- 5c. Fit the Markov Regime-Switching Model ---
# We test models with different numbers of regimes and select the best one based on BIC.
models = {}
for k in range(2, MAX_REGIMES_TO_TEST + 1):
    print(f"Fitting model with {k} regimes...")
    try:
        mod = MarkovRegression(
            endog=endog_voo,
            k_regimes=k,
            trend="c",  # Constant mean term in each regime
            switching_variance=True,
            exog_tvtp=sm.add_constant(exog_final_lagged),  # Time-Varying Transition Probs
        )
        res = mod.fit(search_reps=20)
        # Validate that each regime has a sufficient number of observations
        assigned_regimes = res.smoothed_marginal_probabilities.idxmax(axis=1)
        if (assigned_regimes.value_counts() < MIN_OBS_PER_REGIME).any():
            print(f"  > Model with {k} regimes rejected: a regime had insufficient data.")
            continue
        models[k] = res
        print(f"  > Model with {k} regimes is valid. BIC: {res.bic:.2f}")
    except Exception as e:
        print(f"  > Failed to fit model with {k} regimes: {e}")

if not models:
    raise RuntimeError("No suitable regime-switching models could be fitted.")

# Select the model with the lowest BIC
best_k = min(models, key=lambda k: models[k].bic)
best_model_results = models[best_k]
print(f"\nBest model selected: {best_k} regimes (Lowest BIC = {best_model_results.bic:.2f})")

# --- 5d. Interpret and Label the Regimes ---
# To make regimes interpretable, we sort them by their volatility.
regime_vols = best_model_results.params.filter(like="sigma2").sort_values()
regime_order = regime_vols.index.str.extract(r"\[(\d+)\]")[0].astype(int)
regime_map = {old_idx: new_idx for new_idx, old_idx in enumerate(regime_order)}

# Display characteristics of each sorted regime
sorted_params = pd.DataFrame()
for i in range(best_k):
    original_idx = regime_order.iloc[i]
    # Annualize parameters for interpretability
    mean_ann = best_model_results.params[f"const[{original_idx}]"] * 12 * 100
    vol_ann = np.sqrt(best_model_results.params[f"sigma2[{original_idx}]"]) * np.sqrt(12) * 100
    sorted_params[f"Regime {i}"] = [f"{mean_ann:.1f}%", f"{vol_ann:.1f}%"]
sorted_params.index = ["Annualized Mean (VOO)", "Annualized Volatility (VOO)"]
print("\nCharacteristics of Identified Market Regimes (Sorted by Volatility):")
print(sorted_params)

# Get the final, sorted series of regime probabilities
smoothed_probs = best_model_results.smoothed_marginal_probabilities.rename(
    columns=regime_map
).sort_index(axis=1)
regime_series = smoothed_probs.idxmax(axis=1).rename("regime")


# ==============================================================================
# SECTION 6: REGIME-AWARE DYNAMIC STRATEGY
# ==============================================================================
print("\n--- Section 6: Building and Backtesting the Dynamic Strategy ---")
# --- 6a. Calculate Regime-Specific Optimal Portfolios ---
# We compute a separate optimal portfolio for each identified regime.
regime_frontiers = {}
regime_optimal_weights = {}
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, best_k))

for i in range(best_k):
    print(f"\nAnalyzing Regime {i}...")
    in_regime_periods = regime_series == i
    if in_regime_periods.sum() < max(24, n_assets):
        print(f"  > Skipping Regime {i}, not enough data. Using VOO as fallback.")
        w_fallback = np.array([1.0 if s == "VOO" else 0.0 for s in etf_symbols])
        regime_optimal_weights[i] = w_fallback
        continue

    # Estimate parameters using only data from this regime
    returns_regime = returns_final[in_regime_periods]
    mu_regime = (returns_regime.mean().values * 12) - expense_vector
    cov_regime = LedoitWolf().fit(returns_regime.values).covariance_ * 12

    # Generate the efficient frontier for this regime
    ef_regime = prune_frontier(efficient_frontier(cov_regime, mu_regime, n_points=FRONTIER_POINTS))
    regime_frontiers[i] = ef_regime

    # Find optimal portfolio by matching the overall VOO benchmark's volatility
    _, w_opt = select_portfolio(ef_regime, "sigma", voo_sigma_annual)
    if w_opt is not None:
        regime_optimal_weights[i] = w_opt
        print(f"  > Top 3 ETFs for Regime {i} Portfolio (matching VOO vol):")
        top_indices = np.argsort(w_opt)[-3:][::-1]
        for idx in top_indices:
            if w_opt[idx] > 0.01:
                symbol = etf_symbols[idx]
                name = etf_name_map.get(symbol, "Unknown")
                print(f"    {symbol} ({name}): {w_opt[idx]:.2%}")

        # Plot the frontier and the selected optimal point
        plt.plot(
            ef_regime["sigma"], ef_regime["mu"], label=f"Regime {i} Frontier", color=colors[i], lw=2
        )
        opt_sigma = np.sqrt(w_opt.T @ cov_regime @ w_opt)
        opt_mu = w_opt.T @ mu_regime
        plt.scatter(
            opt_sigma, opt_mu, marker="*", s=250, color=colors[i], zorder=5, edgecolors="black"
        )
    else:
        print("  > Could not find optimal portfolio. Using VOO as fallback.")
        w_fallback = np.array([1.0 if s == "VOO" else 0.0 for s in etf_symbols])
        regime_optimal_weights[i] = w_fallback

# Finalize and show the plot of all regime frontiers
plt.scatter([voo_sigma_annual], [voo_mu_annual], color="black", marker="X", s=200, label="VOO (Overall)", zorder=5)
plt.title("Efficient Frontiers for Each Market Regime", fontsize=16)
plt.xlabel("Annualized Volatility (sigma)", fontsize=12)
plt.ylabel("Annualized Return (mu)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6b. Backtest the Dynamic Strategy ---
# At each month, the portfolio is a blend of the regime-optimal portfolios,
# weighted by the smoothed probability of being in each regime at that time.
dynamic_weights_list = []
for t in range(len(returns_final)):
    probs_t = smoothed_probs.iloc[t]
    blended_w = np.zeros(n_assets)
    # Create blended portfolio by weighting each regime's portfolio by its probability
    for i in range(best_k):
        if i in regime_optimal_weights:
            blended_w += probs_t[i] * regime_optimal_weights[i]
    dynamic_weights_list.append(blended_w / blended_w.sum())

# Calculate the monthly returns of this dynamic portfolio
dynamic_port_returns = np.sum(
    np.array(dynamic_weights_list) * returns_final[etf_symbols].values, axis=1
)
dynamic_returns_series = pd.Series(dynamic_port_returns, index=returns_final.index)


# ==============================================================================
# SECTION 7: FINAL PERFORMANCE COMPARISON
# ==============================================================================
print("\n--- Section 7: Comparing All Strategies ---")


# --- 7a. Define a Performance Metrics Calculator ---
def calculate_performance_metrics(returns_series):
    """
    Calculates key performance metrics for a series of returns.

    Args:
        returns_series (pd.Series): A series of periodic (e.g., monthly) returns.

    Returns:
        dict: A dictionary of performance metrics.
    """
    if returns_series.empty or returns_series.isnull().all():
        return {
            "Total Return (%)": np.nan, "Annualized Return (%)": np.nan,
            "Annualized Volatility (%)": np.nan, "Sharpe Ratio": np.nan,
            "Max Drawdown (%)": np.nan,
        }
    n_periods_per_year = 12  # For monthly returns
    ann_return = ((1 + returns_series.mean()) ** n_periods_per_year - 1) * 100
    ann_vol = returns_series.std() * np.sqrt(n_periods_per_year) * 100
    sharpe = sharpe_ratio(ann_return / 100, ann_vol / 100)
    
    cumulative_returns = (1 + returns_series).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak - 1) * 100
    max_drawdown = drawdown.min()

    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": ann_return,
        "Annualized Volatility (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown,
    }


# --- 7b. Prepare All Strategy Returns for Comparison ---
# Use a common lookback window for a fair comparison
lookback_years = 7
end_date_lookback = max(returns_final.index)
start_date_lookback = end_date_lookback - pd.DateOffset(years=lookback_years)
returns_win = returns_final.loc[start_date_lookback:end_date_lookback].dropna(how="all")

# Helper function to calculate strategy returns safely
def get_strategy_returns(weights):
    if weights is not None:
        return returns_win[etf_symbols] @ weights
    return pd.Series(dtype=float, index=returns_win.index)

strategies = {
    "VOO Benchmark": returns_win["VOO"],
    "Static Raw (Risk-Match)": get_strategy_returns(w_sigma_raw),
    "Static L1 Regularized (Risk-Match)": get_strategy_returns(w_sigma_reg_l1),
    "Static Shrunk (Risk-Match)": get_strategy_returns(w_sigma_shrunk),
    "Static Resampled": get_strategy_returns(w_resampled),
    "Black-Litterman": get_strategy_returns(w_bl_opt),
    "Risk Parity": get_strategy_returns(rp_weights),
    "Hierarchical Risk Parity": get_strategy_returns(hrp_weights),
    "HRP-MVO Blended": get_strategy_returns(w_tilt),
    "DCC-GARCH Dynamic": dynamic_returns_series_dcc.loc[start_date_lookback:end_date_lookback],
    "Regime-Aware Dynamic": dynamic_returns_series.loc[start_date_lookback:end_date_lookback],
}


# --- 7c. Plot Cumulative Performance ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))
for name, rtn in strategies.items():
    if rtn.dropna().empty: continue
    growth = (1 + rtn).cumprod()
    if name.lower().startswith("voo"):
        growth.plot(ax=ax, label=name, lw=3, linestyle="--", color="black")
    else:
        growth.plot(ax=ax, label=name, lw=2)
ax.set_title(f"Cumulative Performance – Last {lookback_years} Years", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Growth of $1 (log scale)", fontsize=12)
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# --- 7d. Display Final Performance Metrics Table ---
all_perf_metrics = {name: calculate_performance_metrics(ret.dropna()) for name, ret in strategies.items()}
all_perf_df = pd.DataFrame(all_perf_metrics).T
print("\n" + "=" * 70)
print("      COMPREHENSIVE STRATEGY PERFORMANCE METRICS")
print("=" * 70)
print(all_perf_df.sort_values(by="Sharpe Ratio", ascending=False))
print("=" * 70 + "\n")


# --- 7e. Visualize Regime Probabilities vs. Market Returns ---
fig, axes = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
)
smoothed_probs.plot(ax=axes[0], kind="area", stacked=True, colormap="viridis", alpha=0.8)
axes[0].set_title("Smoothed Probabilities of Each Market Regime Over Time", fontsize=14)
axes[0].set_ylabel("Probability")
axes[0].legend(title="Regime", loc="upper left")
returns_final["VOO"].plot(ax=axes[1], color="black", label="VOO Monthly Return", alpha=0.7)
axes[1].set_title("VOO Monthly Returns", fontsize=14)
axes[1].set_ylabel("Return")
axes[1].axhline(0, color="grey", lw=1, linestyle="--")
plt.xlabel("Date", fontsize=12)
plt.tight_layout()
plt.show()


# --- 7f. Monte Carlo Simulation ---
# Simulate future returns to see how our portfolios might perform under a wide
# range of possible outcomes, based on the historical return distribution.
print(f"Running Monte Carlo simulation with {MC_SIM_SCENARIOS} scenarios...")
# Use monthly parameters from the full sample for the simulation
monthly_mu_sample = annual_mu / 12
rng = np.random.default_rng(seed=42)  # For reproducibility

# Generate all simulated paths at once for efficiency
simulated_returns_monthly = rng.multivariate_normal(
    mean=monthly_mu_sample,
    cov=sample_cov,  # Use the original sample covariance
    size=(MC_SIM_SCENARIOS, MC_SIM_HORIZON_MONTHS),
)


def simulate_portfolio_performance(weights):
    """Calculates performance metrics from simulated return paths."""
    if weights is None or np.isnan(weights).any():
        return {
            "Mean Ann. Return (%)": np.nan, "Ann. Volatility (%)": np.nan,
            "Sharpe Ratio": np.nan, "VaR 5% (Ann.) (%)": np.nan,
        }
    portfolio_sim_returns = simulated_returns_monthly @ weights
    # Calculate annualized metrics from the simulation results
    mean_monthly_return = np.mean(portfolio_sim_returns)
    std_monthly_return = np.std(portfolio_sim_returns)
    
    annual_mean_return = mean_monthly_return * 12 * 100
    annual_volatility = std_monthly_return * np.sqrt(12) * 100
    sharpe = sharpe_ratio(annual_mean_return/100, annual_volatility/100)
    
    # Value-at-Risk (VaR): The worst expected annualized loss at a 5% confidence level.
    var_5_percent = np.percentile(portfolio_sim_returns, 5) * 12 * 100
    
    return {
        "Mean Ann. Return (%)": annual_mean_return,
        "Ann. Volatility (%)": annual_volatility,
        "Sharpe Ratio": sharpe,
        "VaR 5% (Ann.) (%)": var_5_percent,
    }

# Define weights for all strategies to be simulated
voo_weights = np.array([1.0 if s == "VOO" else 0.0 for s in etf_symbols])
simulation_portfolios = {
    "VOO Benchmark": voo_weights,
    "Static Raw (Risk-Match)": w_sigma_raw,
    "Static L1 Regularized (Risk-Match)": w_sigma_reg_l1,
    "Static Shrunk (Risk-Match)": w_sigma_shrunk,
    "Static Resampled": w_resampled,
    "Black-Litterman": w_bl_opt,
    "Risk Parity": rp_weights,
    "Hierarchical Risk Parity": hrp_weights,
    "HRP-MVO Blended": w_tilt,
}

# Run simulation for each portfolio
sim_results = {name: simulate_portfolio_performance(w) for name, w in simulation_portfolios.items()}
sim_results_df = pd.DataFrame(sim_results).T

print("\n" + "=" * 70)
print(f"      MONTE CARLO SIMULATION SUMMARY ({MC_SIM_HORIZON_MONTHS // 12}-YEAR HORIZON)")
print("=" * 70)
print(sim_results_df.sort_values(by="Sharpe Ratio", ascending=False))
print("=" * 70 + "\n")

print("--- Analysis Complete ---")
