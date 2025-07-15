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

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Optimization & Statistical Modeling Libraries
import cvxopt as opt
import cvxpy as cp
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from arch import arch_model
from arch.univariate import ConstantMean, GARCH
from arch.multivariate import DCC
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from pandas_datareader.data import DataReader

# Web Scraping & Interactive Widget Libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import ipywidgets as widgets
from IPython.display import display

# --- Global Settings & Constants ---
# Set the working directory. The chromedriver executable and any local data
# files should be placed in this directory.
# NOTE: You may need to change this path.
DIRECTORY = '.'
os.chdir(DIRECTORY)

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
# Configure pandas to display floating-point numbers with 3 decimal places for readability.
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
# Configure the CVXOPT solver to not display progress messages during optimization.
opt.solvers.options['show_progress'] = False


# ==============================================================================
# SECTION 2: DATA LOADING & PREPARATION
# ==============================================================================
print("--- Section 2: Loading ETF Data and Calculating Returns ---")

# --- 2a. Load ETF Metadata Directly from Vanguard Website ---
# We use Selenium to scrape the Vanguard advisor site, which lists all ETFs.
# This approach ensures we have an up-to-date list of tickers, names, and
# expense ratios. The site uses JavaScript and pagination, making Selenium ideal.

# Setup Selenium Chrome driver in headless mode (no visible browser window)
chrome_options = Options()
chrome_options.add_argument("--headless=new")       # Use the latest headless mode
chrome_options.add_argument("--disable-gpu")        # Optional: improves stability on some systems
# Point to the local chromedriver binary (must match your installed Chrome version)
chrome_service = Service(executable_path="./chromedriver")
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

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
                fund_name = re.sub(r"\s*(NEW FUND)?\s*$", "", raw_name.replace("\n", " ")).strip()
                
                # Expense ratio is in the 8th column; clean and convert to float
                expense_text = cells[7].text.strip().replace('%', '').strip()
                expense_ratio = float(expense_text) / 100 if expense_text else None
                
                data.append({
                    "Symbol": symbol,
                    "Fund name": fund_name,
                    "Expense ratio": expense_ratio
                })
            except (IndexError, ValueError):
                # Skip row if any element is missing or fails to parse
                continue
    except Exception as e:
        print(f"Error extracting table data: {e}")
    return data

# Scrape data from the paginated ETF table
try:
    print("Scraping ETF data from Vanguard website...")
    # Step 1: Load the Vanguard ETF page and wait for it to render
    driver.get("https://advisors.vanguard.com/investments/etfs")
    time.sleep(6)  # Allow time for JavaScript to load the table

    # Step 2: Extract data from the first page
    all_etf_data = extract_etf_data_from_page(driver)

    # Step 3: Click the "Next page" button to load the remaining ETFs
    try:
        next_button = driver.find_element(By.XPATH, '//button[@aria-label[contains(., "Forward one page")]]')
        next_button.click()
        time.sleep(5)  # Wait for the second page to load

        # Step 4: Extract data from the second page
        all_etf_data += extract_etf_data_from_page(driver)
    except Exception as e:
        print(f"Could not navigate to the second page or it doesn't exist. Reason: {e}")

    # Step 5: Process the scraped data
    df_etf_metadata = pd.DataFrame(all_etf_data)
    etf_name_map = dict(zip(df_etf_metadata['Symbol'], df_etf_metadata['Fund name']))
    etf_expense_map = dict(zip(df_etf_metadata['Symbol'], df_etf_metadata['Expense ratio']))
    etf_symbols = list(etf_name_map.keys())
    print(f"Successfully extracted metadata for {len(etf_symbols)} ETFs.")
    print("Sample of extracted ETF data:")
    print(df_etf_metadata.head())

except Exception as e:
    # If scraping fails, fall back to a predefined list of core ETFs
    print(f"Could not complete web scraping. Reason: {e}")
    print("Falling back to a predefined list of core ETFs.")
    etf_symbols = ['VOO', 'VTI', 'VEA', 'VWO', 'BND', 'BNDX', 'VGIT', 'VGLT', 'VTIP', 'MUB']
    etf_name_map = {s: s for s in etf_symbols}
    etf_expense_map = {s: 0.0003 for s in etf_symbols}  # Use a reasonable default

finally:
    # Always close the browser session
    driver.quit()


# --- 2b. Filter the ETF Universe ---
# To create a diversified portfolio of broad asset classes, we remove
# specialized, sector-specific ETFs and redundant funds.
industry_keywords = [
    'Energy', 'Health Care', 'Consumer', 'Materials', 'Financials',
    'Utilities', 'Real Estate', 'Industrials', 'Communication', 'Information Technology'
]
# List of specific ETFs to remove (often sector-focused or overlapping with broader funds)
remove_symbols = ['VGT', 'VHT', 'VPU', 'VDC', 'VAW', 'VIS', 'VFH', 'VNQ', 'VOX', 'VDE', 'VCR']

def is_industry_or_redundant(symbol, name_map):
    """Checks if an ETF is sector-specific or on the removal list."""
    name = name_map.get(symbol, '')
    is_industry = any(keyword in name for keyword in industry_keywords)
    is_redundant = symbol in remove_symbols
    return is_industry or is_redundant

etf_symbols = [s for s in etf_symbols if not is_industry_or_redundant(s, etf_name_map)]
# Ensure the list contains only unique symbols
etf_symbols = list(dict.fromkeys(etf_symbols))
print(f"\nFiltered down to {len(etf_symbols)} ETFs for analysis.")


# --- 2c. Fetch Historical Price Data ---
def get_total_return_series(ticker):
    """
    Fetches maximum available historical price data for a ticker from Yahoo Finance,
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
        df = stock.history(
            period="max",
            auto_adjust=False,
            back_adjust=True
        )[['Close']].rename(columns={'Close': ticker})
        return df
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        return pd.DataFrame()

# Loop through symbols and combine their price series into one DataFrame
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

# Limit data to the last N years for a more relevant analysis window.
cutoff_date = returns_monthly.index.max() - pd.DateOffset(years=ANALYSIS_YEARS)
returns_monthly = returns_monthly[returns_monthly.index >= cutoff_date]

# Data Cleaning:
# 1. Drop any ETF (column) that lacks a significant number of data points.
min_observations = int(len(returns_monthly) * 0.50)
returns_monthly = returns_monthly.dropna(axis=1, thresh=min_observations)
# 2. Drop any month (row) that still has missing values after column filtering.
returns_monthly = returns_monthly.dropna(axis=0)

# Update the final list of ETF symbols and related data based on the cleaned DataFrame.
etf_symbols = returns_monthly.columns.tolist()

# The S&P 500 ETF (VOO) is our primary benchmark; the analysis requires it.
if 'VOO' not in etf_symbols:
    raise ValueError("VOO data is missing or was dropped. It is required for benchmark comparison.")

# Create a NumPy array of expense ratios in the same order as our final ETF symbols.
expense_vector = np.array([etf_expense_map.get(sym, 0.0) for sym in etf_symbols])

print(f"\nFinal analysis will use {len(etf_symbols)} ETFs over {len(returns_monthly)} months.")
print(f"Analysis period: {returns_monthly.index.min().date()} to {returns_monthly.index.max().date()}")


# ==============================================================================
# SECTION 3: STATIC PORTFOLIO ANALYSIS (MEAN-VARIANCE OPTIMIZATION)
# ==============================================================================
print("\n--- Section 3: Performing Static Mean-Variance Optimization ---")

# --- 3a. Estimate Expected Returns and Covariance ---
# These are the two key inputs for Markowitz portfolio optimization.
# 1. Expected Returns (mu): The anticipated annualized return for each asset.
# 2. Covariance Matrix (Sigma): A measure of how asset returns move together.

# Calculate historical annualized mean returns, net of expense ratios.
# We multiply by 12 to annualize the monthly mean returns.
annual_mu_sample = (returns_monthly.mean().values * 12) - expense_vector
# The sample covariance matrix is calculated from historical returns and annualized.
sample_cov = returns_monthly.cov().values
annual_cov_sample = sample_cov * 12

# --- 3b. Apply Shrinkage to Improve Covariance Estimates ---
# Historical sample estimates can be "noisy" and poor predictors of the future.
# Shrinkage techniques adjust these estimates to be more stable and robust.
# Ledoit-Wolf Shrinkage computes an optimal blend of the sample covariance
# matrix and a more structured, less noisy target matrix. This is a standard
# way to get a more reliable covariance estimate.
lw = LedoitWolf().fit(returns_monthly.values)
annual_cov_shrunk = lw.covariance_ * 12

# We will use the sample returns and the shrunk covariance for our primary model.
annual_mu = annual_mu_sample
print("Annualized Expected Returns (Sample):")
print(pd.Series(annual_mu, index=etf_symbols).round(4))


# --- 3c. Define the Efficient Frontier Optimizer ---
def efficient_frontier(cov_mat, mu_vec, n_points=50):
    """
    Calculates the efficient frontier using the Markowitz model. The "Efficient
    Frontier" is the set of portfolios that provide the highest expected return
    for a given level of risk (volatility).

    This function uses the CVXOPT quadratic programming solver to find portfolio
    weights that minimize variance for a range of target returns.

    Args:
        cov_mat (np.array): The annualized covariance matrix of asset returns.
        mu_vec (np.array): The annualized vector of expected asset returns.
        n_points (int): The number of points to calculate along the frontier.

    Returns:
        dict: A dictionary containing returns ('mu'), volatilities ('sigma'),
              and portfolio weights ('weights') for each point on the frontier.
    """
    n_assets = len(mu_vec)
    # The optimization problem is to minimize: (1/2) * w' * P * w
    # subject to constraints. Here, P is the covariance matrix.
    P = opt.matrix(cov_mat)
    q = opt.matrix(np.zeros((n_assets, 1)))  # No linear term in the objective

    # Constraints:
    # 1. Weights must be non-negative (G*w <= h). We define -w_i <= 0 for each asset.
    # 2. Weights must be <= 1. We define w_i <= 1 for each asset.
    G = opt.matrix(np.vstack([-np.eye(n_assets), np.eye(n_assets)]))
    h = opt.matrix(np.vstack([np.zeros((n_assets, 1)), np.ones((n_assets, 1))]))
    
    # Equality Constraints (A*w = b):
    # 1. Sum of weights must equal 1.
    # 2. Portfolio return must equal the target return.
    A = opt.matrix(np.vstack([mu_vec, np.ones((1, n_assets))]))

    # Iterate through a range of target returns to trace the frontier.
    target_mus = np.linspace(mu_vec.min(), mu_vec.max(), n_points)
    frontier = {'mu': [], 'sigma': [], 'weights': []}

    for mu_target in target_mus:
        b = opt.matrix([mu_target, 1.0])
        try:
            solution = opt.solvers.qp(P, q, G, h, A, b)
            if solution['status'] == 'optimal':
                weights = np.array(solution['x']).flatten()
                sigma = np.sqrt(weights.T @ cov_mat @ weights)
                frontier['mu'].append(mu_target)
                frontier['sigma'].append(sigma)
                frontier['weights'].append(weights)
        except ValueError:
            # Solver may fail if no feasible solution exists for a target return.
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
        frontier (dict): The efficient frontier dictionary from efficient_frontier().
        target_metric (str): The metric to match ('mu' or 'sigma').
        target_value (float): The target return or volatility.

    Returns:
        tuple: Index and weights of the selected portfolio, or (None, None).
    """
    if not frontier[target_metric]:
        return None, None
    diffs = np.abs(np.array(frontier[target_metric]) - target_value)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]


# --- 3e. Generate Frontiers and Select Key Portfolios ---
print("\nGenerating efficient frontiers...")
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
    'Raw (Return-Matched)': w_mu_raw,
    'Raw (Risk-Matched)': w_sigma_raw,
    'Shrunk (Return-Matched)': w_mu_shrunk,
    'Shrunk (Risk-Matched)': w_sigma_shrunk
}

for label, weights in portfolios_to_display.items():
    if weights is not None:
        print(f"\nTop 3 ETFs for {label} Portfolio:")
        # Sort weights in descending order and get the top 3 indices
        top_indices = np.argsort(weights)[-3:][::-1]
        for i in top_indices:
            # Only show assets with a meaningful weight
            if weights[i] > 0.001:
                symbol = etf_symbols[i]
                name = etf_name_map.get(symbol, 'Unknown')
                print(f"  {symbol} ({name}): {weights[i]:.2%}")


# --- 3f. Plot Static Efficient Frontiers ---
plt.figure(figsize=(12, 8))
plt.plot(ef_raw['sigma'], ef_raw['mu'], 'o-', label='Raw Estimate Frontier', alpha=0.7)
plt.plot(ef_shrunk['sigma'], ef_shrunk['mu'], 'o-', label='Shrunk Covariance Frontier', lw=2)
plt.scatter([voo_sigma_annual], [voo_mu_annual], color='red', marker='X', s=200, label='VOO Benchmark', zorder=5)

plt.title('Efficient Frontiers: Raw vs. Shrunk Covariance', fontsize=16)
plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
plt.ylabel('Annualized Expected Return', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()


# ==============================================================================
# SECTION 4: ADVANCED STATIC MODELS & ROBUSTNESS CHECKS
# ==============================================================================
print("\n--- Section 4: Advanced Static Models & Robustness Checks ---")

# --- 4a. Resampled Efficient Frontier (Bootstrapping) ---
# This technique addresses "estimation error" by creating many new return
# datasets via bootstrapping (sampling with replacement). An optimal portfolio is
# found for each bootstrapped sample, and the final portfolio is the average of
# all these optimal portfolios. This typically leads to a more diversified and
# stable allocation that is less sensitive to outliers in the original data.
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
        _, w_boot = select_portfolio(ef_boot, 'sigma', voo_sigma_annual)
        
        if w_boot is not None:
            resampled_weights_list.append(w_boot)
            
    except (ValueError, np.linalg.LinAlgError):
        # Skip iteration if the solver or covariance estimation fails
        continue

# The final resampled portfolio is the average of the weights from all iterations
if resampled_weights_list:
    w_resampled = np.mean(resampled_weights_list, axis=0)
    # MODIFICATION: Changed from Top 5 to Top 3
    print("\nTop 3 ETFs for Resampled Portfolio:")
    top_indices = np.argsort(w_resampled)[-3:][::-1]
    for i in top_indices:
        symbol = etf_symbols[i]
        name = etf_name_map.get(symbol, 'Unknown')
        print(f"  {symbol} ({name}): {w_resampled[i]:.2%}")
else:
    w_resampled = None
    print("\nCould not generate a resampled portfolio.")


# --- 4b. Rolling Window Estimation ---
# This analysis shows how the optimal portfolio allocation would have changed
# over time as new data became available, providing insight into the strategy's stability.
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
        _, w_roll = select_portfolio(ef_roll, 'sigma', voo_sigma_annual)
        
        if w_roll is not None:
            rolling_weights_list.append(pd.Series(w_roll, index=etf_symbols, name=date))
    except (ValueError, np.linalg.LinAlgError):
        continue

# Combine results and plot the weight changes for the most important assets
if rolling_weights_list:
    rolling_weights_df = pd.concat(rolling_weights_list, axis=1).T
    # MODIFICATION: Changed from Top 5 to Top 3
    # Identify the top 3 ETFs by average weight over time
    top_etfs = rolling_weights_df.mean().sort_values(ascending=False).head(3).index
    
    rolling_weights_df[top_etfs].plot(
        figsize=(12, 7),
        title='Top 3 ETF Weights Over Time (Rolling Optimization)',
        lw=2
    )
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--')
    plt.legend(title="ETFs")
    plt.tight_layout()
    plt.show()


# --- 4c. Black-Litterman Optimization ---
# The Black-Litterman model starts with market-implied equilibrium returns and
# then tilts them based on an investor's specific views, creating a blended,
# more intuitive set of expected returns for optimization.
print("\n--- Black-Litterman Portfolio Optimization ---")
# 1. Define the Prior/Equilibrium Returns (pi).
# We use a simple data-driven prior: a shrinkage estimate that blends the
# sample mean with the grand mean of all assets.
neutral_mean = np.full_like(annual_mu_sample, annual_mu_sample.mean())
lambda_shrink = 0.2  # Shrinkage intensity towards the grand mean
pi = lambda_shrink * annual_mu_sample + (1 - lambda_shrink) * neutral_mean

# 2. Specify Investor Views
# View 1: Every other ETF’s expected return is about the same as VOO’s.
# P matrix selects the assets involved in the view.
# Q vector contains the expected outperformance.
# We map symbols to their index in the 'etf_symbols' list.
try:
    voo_idx = etf_symbols.index('VOO')
    rows    = [t for t in etf_symbols if t != "VOO"]

    P = np.zeros((len(rows), n_assets))
    for k, t in enumerate(rows):
        P[k, returns_monthly.columns.get_loc(t)] =  1
        P[k, voo_idx]                       = -1        # relative to VOO

    Q  = np.zeros(len(rows))                            # “≈ 0” differences
 
    # 3. Define Uncertainty in Views (Omega matrix)
    # A diagonal matrix where smaller values mean higher confidence in the view.
    omega = np.diag(np.full(len(Q), 0.005))             # big ⇒ soft view
    
    # 4. Combine Priors and Views to get Posterior Returns (m_bl)
    # The 'tau' parameter scales the uncertainty of the prior.
    bl_tau = 0.05
    cov_inv = np.linalg.inv(bl_tau * annual_cov_shrunk)
    omega_inv = np.linalg.inv(omega)
    
    # Core Black-Litterman formula for the posterior mean
    M_inv = np.linalg.inv(cov_inv + P.T @ omega_inv @ P)
    m_bl = M_inv @ (cov_inv @ pi + P.T @ omega_inv @ Q)

    # 5. Optimize the portfolio using the new Black-Litterman expected returns.
    w_bl = cp.Variable(n_assets)
    # Risk aversion parameter (gamma) implicitly set to 0.5 in the objective
    objective = cp.Maximize(m_bl @ w_bl - 0.5 * cp.quad_form(w_bl, annual_cov_shrunk))
    constraints = [cp.sum(w_bl) == 1, w_bl >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    w_bl_opt = w_bl.value
    
        
    print("Optimal Black-Litterman Portfolio (all weights > 1%):")
    for i, weight in enumerate(w_bl_opt):
        if weight > 0.01:
            print(f"  - {etf_symbols[i]}: {weight:.1%}")
            
    # MODIFICATION: Added "Top 3 ETFs" summary
    if w_bl_opt is not None:
        print("\nTop 3 ETFs for Black-Litterman Portfolio:")
        top_indices = np.argsort(w_bl_opt)[-3:][::-1]
        for i in top_indices:
            symbol = etf_symbols[i]
            name = etf_name_map.get(symbol, 'Unknown')
            print(f"  {symbol} ({name}): {w_bl_opt[i]:.2%}")

except (ValueError, IndexError):
    print("Could not perform Black-Litterman: VEA or VWO not in the asset list.")
    w_bl_opt = None


# --- 4d. Risk Parity Optimization ---
# Risk Parity aims to construct a portfolio where each asset contributes equally
# to the total portfolio risk. Unlike MVO, it ignores expected returns and
# focuses solely on diversifying risk.
print("\n--- Risk Parity Portfolio Optimization ---")
def portfolio_volatility(weights, cov_matrix):
    """Calculates the annualized volatility of a portfolio."""
    return np.sqrt(weights.T @ cov_matrix @ weights)

def risk_contributions(weights, cov_matrix):
    """Calculates each asset's percentage contribution to total portfolio risk."""
    port_vol = portfolio_volatility(weights, cov_matrix)
    if port_vol == 0: return np.zeros_like(weights)
    # Marginal Risk Contribution (MRC)
    mrc = (cov_matrix @ weights) / port_vol
    # Total Risk Contribution = weight * MRC
    return weights * mrc

def risk_parity_objective(weights, cov_matrix):
    """
    Objective function for the optimizer. It seeks to minimize the
    variance of risk contributions across all assets, forcing them to be equal.
    """
    # We want risk contributions to be equal, so we calculate total risk contribution per asset
    total_risk_contributions = risk_contributions(weights, cov_matrix)
    # The target is an equal contribution from each asset
    target_contribution = total_risk_contributions.sum() / len(weights)
    # Minimize the squared differences from this target
    return np.sum((total_risk_contributions - target_contribution)**2)

# --- Solve the optimization problem ---
initial_weights = np.ones(n_assets) / n_assets  # Start with equal weights
bounds = tuple((0.0, 1.0) for _ in range(n_assets))
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

result = minimize(
    fun=risk_parity_objective,
    x0=initial_weights,
    args=(annual_cov_sample,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'disp': False}
)
rp_weights = result.x if result.success else None

if rp_weights is not None:
    print("Optimal Risk Parity Portfolio (all weights > 1%):")
    for i, weight in enumerate(rp_weights):
        if weight > 0.01:
            print(f"  - {etf_symbols[i]}: {weight:.1%}")

    # MODIFICATION: Added "Top 3 ETFs" summary
    print("\nTop 3 ETFs for Risk Parity Portfolio:")
    top_indices = np.argsort(rp_weights)[-3:][::-1]
    for i in top_indices:
        symbol = etf_symbols[i]
        name = etf_name_map.get(symbol, 'Unknown')
        print(f"  {symbol} ({name}): {rp_weights[i]:.2%}")
else:
    print("Risk Parity optimization failed.")


# --- 4e. Hierarchical Risk Parity (HRP) Optimization ---
# HRP is a novel approach that uses graph theory and machine learning to build
# a diversified portfolio. It works in three steps:
# 1. Tree Clustering: Groups similar assets based on their correlation.
# 2. Quasi-Diagonalization: Reorders the covariance matrix based on the hierarchy.
# 3. Recursive Bisection: Distributes weights top-down through the hierarchy.
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
    return inv_var_weights @ sub_cov @ inv_var_weights

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
        weights[right_cluster] *= (1.0 - alpha)
        
        # Add the new sub-clusters to the list to be processed
        clusters.extend([left_cluster, right_cluster])
        
    return weights.sort_index()

# Step 1: Hierarchical Clustering
corr_matrix = returns_monthly.corr()
dist_matrix = correlation_to_distance(corr_matrix)
linkage_matrix = linkage(squareform(dist_matrix), method='single')

# Step 2: Quasi-Diagonalization (Seriation)
# This reorders the assets to place similar assets next to each other.
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

print("Optimal HRP Portfolio (all weights > 1%):")
for i, weight in enumerate(hrp_weights):
    if weight > 0.01:
        print(f"  - {etf_symbols[i]}: {weight:.1%}")

# MODIFICATION: Added "Top 3 ETFs" summary
if hrp_weights is not None:
    print("\nTop 3 ETFs for HRP Portfolio:")
    top_indices = np.argsort(hrp_weights)[-3:][::-1]
    for i in top_indices:
        symbol = etf_symbols[i]
        name = etf_name_map.get(symbol, 'Unknown')
        print(f"  {symbol} ({name}): {hrp_weights[i]:.2%}")

# Optional: Visualize the asset hierarchy with a dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=[etf_symbols[i] for i in sorted_indices], leaf_rotation=90)
plt.title("HRP Asset Clustering (Dendrogram)", fontsize=16)
plt.tight_layout()
plt.show()


# --- 4f. DCC-GARCH Portfolio Optimization ---
# Dynamic Conditional Correlation (DCC) GARCH models are advanced time-series
# models that capture time-varying volatility and correlation. This is useful
# as correlations often spike during market crises.
print("\n--- Rolling DCC-GARCH Optimization ---")
# Step 1: Prepare weekly data for better GARCH model estimation
start_date = returns_monthly.index.min()
try:
    price_weekly = yf.download(etf_symbols, start=start_date, interval='1wk', auto_adjust=True)['Adj Close'].dropna()
    returns_weekly = np.log(price_weekly / price_weekly.shift(1)).dropna()

    # MODIFICATION: Restored GARCH effects testing routine
    print("\n--- Testing for GARCH(1,1) Effects in Weekly Returns ---")
    for symbol in etf_symbols:
        series = returns_weekly[symbol].dropna()
        if len(series) < 20: continue # Not enough data for a meaningful test
        
        # ARCH LM Test to check for autoregressive conditional heteroskedasticity
        lm_test = het_arch(series, nlags=12)
        pval = lm_test[1]
        
        print(f"{symbol}: ARCH LM Test p-value = {pval:.4f}", end="")
        if pval < 0.05:
            print("  → ARCH effects detected (GARCH model is appropriate)")
        else:
            print("  → No significant ARCH effects detected")

    # Step 2: Set up the rolling optimization
    month_ends = returns_weekly.resample('M').last().index
    voo_sigma_annual_weekly = returns_weekly['VOO'].std() * np.sqrt(52)
    rolling_dcc_weights_list = []

    # Step 3: Perform rolling DCC optimization
    for date in month_ends:
        data_window = returns_weekly.loc[:date]
        if len(data_window) < 104: continue # Need enough data to fit the model

        try:
            print(f"Optimizing DCC portfolio for month-end {date.date()}...")
            # Fit univariate GARCH(1,1) models for each asset
            garch_models = [ConstantMean(data_window[s], GARCH(1,1)).fit(disp='off') for s in etf_symbols]
            
            # Fit the DCC model on the standardized residuals of the GARCH models
            dcc_model = DCC(garch_models)
            dcc_res = dcc_model.fit(disp='off')
            
            # Forecast 1-week-ahead covariance and annualize it
            forecasted_cov = dcc_res.forecast(horizon=1).cov.iloc[-1].values * 52
            forecasted_mu = data_window.mean().values * 52
            
            # Find the optimal portfolio using the forecasted parameters
            ef_dcc = efficient_frontier(forecasted_cov, forecasted_mu, n_points=30)
            _, w_dcc = select_portfolio(ef_dcc, 'sigma', voo_sigma_annual_weekly)
            
            if w_dcc is not None:
                rolling_dcc_weights_list.append(pd.Series(w_dcc, index=etf_symbols, name=date))

        except Exception as e:
            print(f"  Skipped {date.date()} due to error: {e}")

    # Step 4: Process and store results
    if rolling_dcc_weights_list:
        dcc_weights_df = pd.concat(rolling_dcc_weights_list, axis=1).T
        
        # MODIFICATION: Added rolling weights plot for DCC-GARCH
        top_dcc_etfs = dcc_weights_df.mean().sort_values(ascending=False).head(3).index
        dcc_weights_df[top_dcc_etfs].plot(
            figsize=(12, 7),
            title='Top 3 ETF Weights Over Time (DCC-GARCH Optimization)',
            lw=2
        )
        plt.ylabel("Portfolio Weight")
        plt.xlabel("Date")
        plt.grid(True, linestyle='--')
        plt.legend(title="ETFs")
        plt.tight_layout()
        plt.show()

        # Create a monthly return series for the DCC strategy
        # Shift weights by 1 to avoid lookahead bias (use this month's weights for next month's return)
        dcc_shifted_weights = dcc_weights_df.shift(1).reindex(returns_monthly.index).ffill()
        
        # Align weights and returns
        common_idx = dcc_shifted_weights.dropna().index.intersection(returns_monthly.index)
        aligned_weights = dcc_shifted_weights.loc[common_idx]
        aligned_returns = returns_monthly.loc[common_idx]
        
        dynamic_returns_series_dcc = pd.Series(
            np.sum(aligned_weights.values * aligned_returns[etf_symbols].values, axis=1),
            index=common_idx
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
# Markets do not behave uniformly; they switch between states like "bull", "bear",
# or "volatile". We use a Markov Switching Model to identify these regimes based
# on market returns (VOO) and external economic indicators.

# --- 5a. Load Exogenous Economic Data ---
# We use VIX (volatility index) and the Treasury yield spread as indicators of
# the broader economic environment.
start_date = returns_monthly.index.min()
end_date = returns_monthly.index.max()

def get_fred_data(start, end):
    """Fetches US Treasury yield and VIX data."""
    print("Fetching VIX and yield curve data from FRED...")
    symbols = {'3M': "DGS3MO", '10Y': "DGS10", 'VIX': "VIXCLS"}
    try:
        df = DataReader(list(symbols.values()), 'fred', start, end)
        df = df.rename(columns={v: k for k, v in symbols.items()})
        df[['3M', '10Y']] = df[['3M', '10Y']] / 100.0  # Convert from percent to decimal
        df['Spread_10Y_3M'] = df['10Y'] - df['3M']
        return df.dropna()
    except Exception as e:
        print(f"Could not fetch FRED data: {e}")
        return pd.DataFrame()

exog_df = get_fred_data(start_date, end_date)

# --- 5b. Align and Prepare Data for Modeling ---
# Align all data to our monthly return frequency.
exog_monthly = exog_df.resample('M').last().ffill()

# Find the common date range between returns and economic data.
common_index = returns_monthly.index.intersection(exog_monthly.index)
returns_aligned = returns_monthly.loc[common_index]
exog_aligned = exog_monthly.loc[common_index]

# We use LAGGED economic data to predict the NEXT month's regime.
# This is crucial to prevent lookahead bias.
exog_lagged = exog_aligned.shift(1).dropna()

# Final alignment after lagging.
final_index = returns_aligned.index.intersection(exog_lagged.index)
returns_final = returns_aligned.loc[final_index]
exog_final_lagged = exog_lagged.loc[final_index, ['VIX', 'Spread_10Y_3M']]

# The model identifies regimes based on the returns of the broad market (VOO).
endog_voo = returns_final['VOO']
print(f"Final dataset for regime modeling has {len(endog_voo)} monthly observations.")

# --- 5c. Fit the Markov Regime-Switching Model ---
# We test models with different numbers of regimes (2, 3, 4) and select the
# best one based on the Bayesian Information Criterion (BIC), which balances
# model fit with complexity.
models = {}
for k in range(2, MAX_REGIMES_TO_TEST + 1):
    print(f"Fitting model with {k} regimes...")
    try:
        # This model allows both mean return and volatility to be different in each regime.
        # 'exog_tvtp' allows the economic data to influence the probability of switching regimes.
        mod = MarkovRegression(
            endog=endog_voo,
            k_regimes=k,
            trend='c',  # Constant mean term in each regime
            switching_variance=True,
            exog_tvtp=sm.add_constant(exog_final_lagged) # Time-Varying Transition Probabilities
        )
        res = mod.fit(search_reps=20) # Search for the best starting parameters
        
        # Validation: Ensure each regime has a sufficient number of observations.
        assigned_regimes = res.smoothed_marginal_probabilities.idxmax(axis=1)
        if (assigned_regimes.value_counts() < MIN_OBS_PER_REGIME).any():
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
# To make the regimes interpretable, we sort them by their volatility.
# This gives consistent labels, e.g., "Regime 0" is always the lowest volatility state.
regime_vols = best_model_results.params.filter(like='sigma2').sort_values()
regime_order = regime_vols.index.str.extract(r'\[(\d+)\]')[0].astype(int)
regime_map = {old_idx: new_idx for new_idx, old_idx in enumerate(regime_order)}

# Display the characteristics (mean return and volatility) of each sorted regime.
sorted_params = pd.DataFrame()
for i in range(best_k):
    original_idx = regime_order.iloc[i]
    # Annualize parameters for interpretability
    mean_ann = best_model_results.params[f'const[{original_idx}]'] * 12 * 100
    vol_ann = np.sqrt(best_model_results.params[f'sigma2[{original_idx}]']) * np.sqrt(12) * 100
    sorted_params[f'Regime {i}'] = [f'{mean_ann:.1f}%', f'{vol_ann:.1f}%']

sorted_params.index = ['Annualized Mean (VOO)', 'Annualized Volatility (VOO)']
print("\nCharacteristics of Identified Market Regimes (Sorted by Volatility):")
print(sorted_params)

# Get the final, sorted series of regime probabilities.
smoothed_probs = best_model_results.smoothed_marginal_probabilities.rename(columns=regime_map).sort_index(axis=1)
regime_series = smoothed_probs.idxmax(axis=1).rename('regime')


# ==============================================================================
# SECTION 6: REGIME-AWARE DYNAMIC STRATEGY
# ==============================================================================
print("\n--- Section 6: Building and Backtesting the Dynamic Strategy ---")

# --- 6a. Calculate Regime-Specific Optimal Portfolios ---
# We now compute a separate optimal portfolio for each identified regime. The goal
# is to hold the best possible portfolio for the current market environment.
regime_frontiers = {}
regime_optimal_weights = {}

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, best_k))

for i in range(best_k):
    print(f"\nAnalyzing Regime {i}...")
    in_regime_periods = (regime_series == i)
    
    # We need enough data points in a regime to get reliable estimates.
    if in_regime_periods.sum() < max(24, n_assets):
        print(f"  > Skipping Regime {i}, not enough data ({in_regime_periods.sum()} points). Using VOO fallback.")
        # Fallback to a 100% VOO portfolio for this regime if we can't optimize.
        w_fallback = np.array([1.0 if s == 'VOO' else 0.0 for s in etf_symbols])
        regime_optimal_weights[i] = w_fallback
        continue

    # Estimate parameters using only the data from this regime.
    returns_regime = returns_final[in_regime_periods]
    mu_regime = (returns_regime.mean().values * 12) - expense_vector
    cov_regime = LedoitWolf().fit(returns_regime.values).covariance_ * 12
    
    # Generate the efficient frontier for this regime.
    ef_regime = efficient_frontier(cov_regime, mu_regime, n_points=FRONTIER_POINTS)
    regime_frontiers[i] = ef_regime
    
    # Find the optimal portfolio by matching the overall VOO benchmark's volatility.
    _, w_opt = select_portfolio(ef_regime, 'sigma', voo_sigma_annual)
    
    if w_opt is not None:
        regime_optimal_weights[i] = w_opt
        # MODIFICATION: Changed from Top 5 to Top 3
        print(f"  > Top 3 ETFs for Regime {i} Portfolio (matching VOO vol):")
        top_indices = np.argsort(w_opt)[-3:][::-1]
        for idx in top_indices:
            if w_opt[idx] > 0.01:
                symbol = etf_symbols[idx]
                name = etf_name_map.get(symbol, 'Unknown')
                print(f"    {symbol} ({name}): {w_opt[idx]:.2%}")
        
        # Plot the frontier and the selected optimal point.
        plt.plot(ef_regime['sigma'], ef_regime['mu'], label=f'Regime {i} Frontier', color=colors[i], lw=2)
        opt_sigma = np.sqrt(w_opt.T @ cov_regime @ w_opt)
        opt_mu = w_opt.T @ mu_regime
        plt.scatter(opt_sigma, opt_mu, marker='*', s=250, color=colors[i], zorder=5, edgecolors='black')
    else:
        print("  > Could not find optimal portfolio. Using VOO as fallback.")
        w_fallback = np.array([1.0 if s == 'VOO' else 0.0 for s in etf_symbols])
        regime_optimal_weights[i] = w_fallback

# Finalize and show the plot of all regime frontiers.
plt.scatter([voo_sigma_annual], [voo_mu_annual], color='black', marker='X', s=200, label='VOO (Overall)', zorder=5)
plt.title('Efficient Frontiers for Each Market Regime', fontsize=16)
plt.xlabel('Annualized Volatility (Sigma)', fontsize=12)
plt.ylabel('Annualized Return (Mu)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6b. Backtest the Dynamic Strategy ---
# At each month, our portfolio is a blend of the regime-optimal portfolios,
# weighted by the smoothed probability of being in each regime at that time.
dynamic_weights_list = []
for t in range(len(returns_final)):
    # Get the smoothed probabilities for this time step.
    probs_t = smoothed_probs.iloc[t]
    blended_w = np.zeros(n_assets)
    
    # Create the blended portfolio by weighting each regime's optimal portfolio by its probability.
    for i in range(best_k):
        if i in regime_optimal_weights:
            blended_w += probs_t[i] * regime_optimal_weights[i]
            
    # Normalize weights to ensure they sum to 1.
    dynamic_weights_list.append(blended_w / blended_w.sum())

# Calculate the monthly returns of this dynamic portfolio.
dynamic_port_returns = np.sum(np.array(dynamic_weights_list) * returns_final[etf_symbols].values, axis=1)
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
            "Max Drawdown (%)": np.nan
        }
        
    # Annualized Return (Geometric)
    n_periods_per_year = 12 # For monthly returns
    ann_return = ((1 + returns_series.mean()) ** n_periods_per_year - 1) * 100
    
    # Annualized Volatility
    ann_vol = returns_series.std() * np.sqrt(n_periods_per_year) * 100
    
    # Sharpe Ratio (assumes risk-free rate is 0)
    # Measures risk-adjusted return. Higher is better.
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan
    
    # Cumulative returns and drawdown
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
        "Max Drawdown (%)": max_drawdown
    }

# --- 7b. Prepare All Strategy Returns for Comparison ---
# Calculate the historical returns for each static portfolio to compare them
# against the dynamic strategies and the VOO benchmark.
strategies = {
    'VOO Benchmark': returns_final['VOO'],
    'Static Raw (Risk-Match)': (returns_final[etf_symbols] @ w_sigma_raw) if w_sigma_raw is not None else pd.Series(dtype=float),
    'Static Shrunk (Risk-Match)': (returns_final[etf_symbols] @ w_sigma_shrunk) if w_sigma_shrunk is not None else pd.Series(dtype=float),
    'Static Resampled': (returns_final[etf_symbols] @ w_resampled) if w_resampled is not None else pd.Series(dtype=float),
    'Black-Litterman': (returns_final[etf_symbols] @ w_bl_opt) if w_bl_opt is not None else pd.Series(dtype=float),
    'Risk Parity': (returns_final[etf_symbols] @ rp_weights) if rp_weights is not None else pd.Series(dtype=float),
    'Hierarchical Risk Parity': (returns_final[etf_symbols] @ hrp_weights) if hrp_weights is not None else pd.Series(dtype=float),
    'DCC-GARCH Dynamic': dynamic_returns_series_dcc,
    'Regime-Aware Dynamic': dynamic_returns_series
}

# --- 7c. Plot Cumulative Performance and Display Metrics Table ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

for name, returns in strategies.items():
    if not returns.dropna().empty:
        # Plot cumulative growth of $1
        (1 + returns).cumprod().plot(ax=ax, label=name, lw=2)

ax.set_title('Cumulative Performance Comparison of All Strategies', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Growth of $1 (Log Scale)', fontsize=12)
ax.set_yscale('log') # Log scale is useful for comparing long-term growth rates.
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# --- 7d. Display Final Performance Metrics Table ---
all_perf_metrics = {name: calculate_performance_metrics(ret.dropna()) for name, ret in strategies.items()}
all_perf_df = pd.DataFrame(all_perf_metrics).T

print("\n" + "="*70)
print("      COMPREHENSIVE STRATEGY PERFORMANCE METRICS")
print("="*70)
print(all_perf_df.sort_values(by='Sharpe Ratio', ascending=False))
print("="*70 + "\n")

# --- 7e. Visualize Regime Probabilities vs. Market Returns ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

# Top plot: Smoothed probabilities of each regime over time
smoothed_probs.plot(ax=axes[0], kind='area', stacked=True, colormap='viridis', alpha=0.8)
axes[0].set_title('Smoothed Probabilities of Each Market Regime Over Time', fontsize=14)
axes[0].set_ylabel('Probability')
axes[0].legend(title='Regime', loc='upper left')

# Bottom plot: VOO monthly returns for context
returns_final['VOO'].plot(ax=axes[1], color='black', label='VOO Monthly Return', alpha=0.7)
axes[1].set_title('VOO Monthly Returns', fontsize=14)
axes[1].set_ylabel('Return')
axes[1].axhline(0, color='grey', lw=1, linestyle='--')

plt.xlabel("Date", fontsize=12)
plt.tight_layout()
plt.show()

# --- 7f. Monte Carlo Simulation ---
# Simulate future returns to see how our portfolios might perform under a wide
# range of possible outcomes, based on the historical return distribution.
print(f"Running Monte Carlo simulation with {MC_SIM_SCENARIOS} scenarios...")
# Use monthly parameters from the full sample for the simulation
monthly_mu_sample = annual_mu / 12
rng = np.random.default_rng(seed=42) # For reproducibility

# Generate all simulated paths at once for efficiency
simulated_returns_monthly = rng.multivariate_normal(
    mean=monthly_mu_sample,
    cov=sample_cov, # Use the original sample covariance
    size=(MC_SIM_SCENARIOS, MC_SIM_HORIZON_MONTHS)
)

def simulate_portfolio_performance(weights):
    """Calculates performance metrics from simulated return paths."""
    if weights is None or np.isnan(weights).any():
        return {'Mean Ann. Return (%)': np.nan, 'Ann. Volatility (%)': np.nan, 'VaR 5% (Ann.) (%)': np.nan}
    
    # Calculate portfolio returns for each scenario and time step
    portfolio_sim_returns = simulated_returns_monthly @ weights
    
    # Calculate annualized metrics from the simulation results
    # Mean of all scenario outcomes
    annual_mean_return = np.mean(portfolio_sim_returns) * 12 * 100
    # Std dev of all scenario outcomes
    annual_volatility = np.std(portfolio_sim_returns) * np.sqrt(12) * 100
    # Value-at-Risk (VaR): The worst expected loss at a 5% confidence level.
    var_5_percent = np.percentile(portfolio_sim_returns, 5) * 12 * 100
    
    return {
        'Mean Ann. Return (%)': annual_mean_return,
        'Ann. Volatility (%)': annual_volatility,
        'VaR 5% (Ann.) (%)': var_5_percent
    }

# Define weights for all strategies to be simulated
voo_weights = np.array([1.0 if s == 'VOO' else 0.0 for s in etf_symbols])
simulation_portfolios = {
    'VOO Benchmark': voo_weights,
    'Static Shrunk (Risk-Match)': w_sigma_shrunk,
    'Static Resampled': w_resampled,
    'Black-Litterman': w_bl_opt,
    'Risk Parity': rp_weights,
    'Hierarchical Risk Parity': hrp_weights
}

# Run simulation for each portfolio
sim_results = {name: simulate_portfolio_performance(w) for name, w in simulation_portfolios.items()}
sim_results_df = pd.DataFrame(sim_results).T

print("\n" + "="*70)
print("      MONTE CARLO SIMULATION SUMMARY (10-YEAR HORIZON)")
print("="*70)
print(sim_results_df)
print("="*70 + "\n")

print("--- Analysis Complete ---")
