# ==============================================================================
#
#        PORTFOLIO OPTIMIZATION & DYNAMIC ASSET ALLOCATION
#                  (WITH OUT-OF-SAMPLE TESTING)
#
# ==============================================================================
#
# OVERVIEW:
#
# This script now performs a rigorous out-of-sample (OOS) backtest.
#
# 1. DATA SPLIT: The historical data is split into two periods:
#    - IN-SAMPLE (Training): All data EXCEPT the last 3 years. All models
#      (MVO, HRP, Regime-Switching, etc.) are trained, and their parameters
#      and optimal weights are determined using ONLY this data.
#    - OUT-OF-SAMPLE (Testing): The final 3 years of data. The performance
#      of the pre-trained models is evaluated on this unseen data to simulate
#      real-world performance.
#
# 2. ANALYSIS: All performance metrics and charts in the final section now
#    reflect the strategy's OOS performance, providing a more realistic
#    assessment of its viability.
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
DIRECTORY = '.'
os.chdir(DIRECTORY)

ANALYSIS_YEARS = 15
# OOS NOTE: Define the out-of-sample period length in years.
OOS_YEARS = 3

# Optimization & Simulation Parameters
FRONTIER_POINTS = 50
RESAMPLE_ITERATIONS = 100
ROLLING_WINDOW_MONTHS = 60

# Regime Modeling Parameters
MIN_OBS_PER_REGIME = 10
MAX_REGIMES_TO_TEST = 4

# --- Initial Setup ---
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
opt.solvers.options['show_progress'] = False


# ==============================================================================
# SECTION 2: DATA LOADING, PREPARATION & SPLIT
# ==============================================================================
print("--- Section 2: Loading ETF Data, Calculating Returns, and Splitting Data ---")

# --- 2a. Load ETF Metadata Directly from Vanguard Website ---
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_service = Service(executable_path="./chromedriver")
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

def extract_etf_data_from_page(driver_instance):
    data = []
    try:
        table_body = driver_instance.find_element(By.CSS_SELECTOR, "table tbody")
        rows = table_body.find_elements(By.TAG_NAME, "tr")
        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                symbol = cells[0].text.strip()
                raw_name = row.find_elements(By.TAG_NAME, "div")[1].text.strip()
                fund_name = re.sub(r"\s*(NEW FUND)?\s*$", "", raw_name.replace("\n", " ")).strip()
                expense_text = cells[7].text.strip().replace('%', '').strip()
                expense_ratio = float(expense_text) / 100 if expense_text else None
                data.append({"Symbol": symbol, "Fund name": fund_name, "Expense ratio": expense_ratio})
            except (IndexError, ValueError):
                continue
    except Exception as e:
        print(f"Error extracting table data: {e}")
    return data
try:
    print("Scraping ETF data from Vanguard website...")
    driver.get("https://advisors.vanguard.com/investments/etfs")
    time.sleep(6)
    all_etf_data = extract_etf_data_from_page(driver)
    try:
        next_button = driver.find_element(By.XPATH, '//button[@aria-label[contains(., "Forward one page")]]')
        next_button.click()
        time.sleep(5)
        all_etf_data += extract_etf_data_from_page(driver)
    except Exception as e:
        print(f"Could not navigate to the second page. Reason: {e}")
    df_etf_metadata = pd.DataFrame(all_etf_data)
    etf_name_map = dict(zip(df_etf_metadata['Symbol'], df_etf_metadata['Fund name']))
    etf_expense_map = dict(zip(df_etf_metadata['Symbol'], df_etf_metadata['Expense ratio']))
    etf_symbols = list(etf_name_map.keys())
    print(f"Successfully extracted metadata for {len(etf_symbols)} ETFs.")
except Exception as e:
    print(f"Web scraping failed. Reason: {e}. Falling back to a predefined list.")
    etf_symbols = ['VOO', 'VTI', 'VEA', 'VWO', 'BND', 'BNDX', 'VGIT', 'VGLT', 'VTIP', 'MUB']
    etf_name_map = {s: s for s in etf_symbols}
    etf_expense_map = {s: 0.0003 for s in etf_symbols}
finally:
    driver.quit()

# --- 2b. Filter the ETF Universe ---
industry_keywords = ['Energy', 'Health Care', 'Consumer', 'Materials', 'Financials', 'Utilities', 'Real Estate', 'Industrials', 'Communication', 'Information Technology']
remove_symbols = ['VGT', 'VHT', 'VPU', 'VDC', 'VAW', 'VIS', 'VFH', 'VNQ', 'VOX', 'VDE', 'VCR']
def is_industry_or_redundant(symbol, name_map):
    name = name_map.get(symbol, '')
    return any(keyword in name for keyword in industry_keywords) or symbol in remove_symbols
etf_symbols = [s for s in etf_symbols if not is_industry_or_redundant(s, etf_name_map)]
etf_symbols = list(dict.fromkeys(etf_symbols))
print(f"\nFiltered down to {len(etf_symbols)} ETFs for analysis.")

# --- 2c. Fetch Historical Price Data ---
all_prices = pd.DataFrame()
for ticker in etf_symbols:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="max", auto_adjust=False, back_adjust=True)[['Close']].rename(columns={'Close': ticker})
        if not df.empty:
            all_prices = pd.concat([all_prices, df], axis=1)
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        continue

# --- 2d. Process and Clean Return Data ---
all_prices.index = pd.to_datetime(all_prices.index).tz_localize(None)
returns_monthly = all_prices.resample('M').last().pct_change()
cutoff_date = returns_monthly.index.max() - pd.DateOffset(years=ANALYSIS_YEARS)
returns_monthly = returns_monthly[returns_monthly.index >= cutoff_date]
min_observations = int(len(returns_monthly) * 0.50)
returns_monthly = returns_monthly.dropna(axis=1, thresh=min_observations)
returns_monthly = returns_monthly.dropna(axis=0)
etf_symbols = returns_monthly.columns.tolist()
if 'VOO' not in etf_symbols:
    raise ValueError("VOO data is missing or was dropped.")
expense_vector = np.array([etf_expense_map.get(sym, 0.0) for sym in etf_symbols])
print(f"\nFull analysis dataset has {len(etf_symbols)} ETFs over {len(returns_monthly)} months.")

# --- 2e. OOS NOTE: Split Data into In-Sample and Out-of-Sample Periods ---
oos_start_date = returns_monthly.index.max() - pd.DateOffset(years=OOS_YEARS) + pd.DateOffset(days=1)
returns_in_sample = returns_monthly[returns_monthly.index < oos_start_date]
returns_out_of_sample = returns_monthly[returns_monthly.index >= oos_start_date]

print(f"\nData has been split for Out-of-Sample testing:")
print(f"  - In-Sample (Training) Period:   {returns_in_sample.index.min().date()} to {returns_in_sample.index.max().date()}")
print(f"  - Out-of-Sample (Testing) Period: {returns_out_of_sample.index.min().date()} to {returns_out_of_sample.index.max().date()}")


# ==============================================================================
# SECTIONS 3 & 4: MODEL FITTING ON **IN-SAMPLE** DATA
# ==============================================================================
# OOS NOTE: All calculations for model parameters and optimal weights
# will now use `returns_in_sample` ONLY. These weights will be treated as fixed
# for the out-of-sample test.

print("\n--- Sections 3 & 4: Fitting All Models on IN-SAMPLE Data ---")

# --- Define Helper Functions ---
# (These are needed for both in-sample fitting and later parts of the script)
def efficient_frontier(cov_mat, mu_vec, n_points=50):
    n_assets = len(mu_vec)
    P = opt.matrix(cov_mat)
    q = opt.matrix(np.zeros((n_assets, 1)))
    G = opt.matrix(np.vstack([-np.eye(n_assets), np.eye(n_assets)]))
    h = opt.matrix(np.vstack([np.zeros((n_assets, 1)), np.ones((n_assets, 1))]))
    A = opt.matrix(np.vstack([mu_vec, np.ones((1, n_assets))]))
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
            pass
    return frontier

def select_portfolio(frontier, target_metric, target_value):
    if not frontier[target_metric]: return None, None
    diffs = np.abs(np.array(frontier[target_metric]) - target_value)
    return diffs.argmin(), frontier['weights'][diffs.argmin()]

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def risk_contributions(weights, cov_matrix):
    port_vol = portfolio_volatility(weights, cov_matrix)
    if port_vol == 0: return np.zeros_like(weights)
    return weights * ((cov_matrix @ weights) / port_vol)

def risk_parity_objective(weights, cov_matrix):
    total_risk_contributions = risk_contributions(weights, cov_matrix)
    target_contribution = total_risk_contributions.sum() / len(weights)
    return np.sum((total_risk_contributions - target_contribution)**2)

def correlation_to_distance(corr_matrix):
    return np.sqrt(0.5 * (1 - corr_matrix))

def get_cluster_variance(cov_matrix, cluster_indices):
    sub_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
    inv_var_weights = 1.0 / np.diag(sub_cov)
    inv_var_weights /= inv_var_weights.sum()
    return inv_var_weights @ sub_cov @ inv_var_weights

def recursive_bisection(cov_matrix, sorted_indices):
    weights = pd.Series(1.0, index=sorted_indices)
    clusters = [sorted_indices]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1: continue
        split_point = len(cluster) // 2
        left_cluster, right_cluster = cluster[:split_point], cluster[split_point:]
        var_left = get_cluster_variance(cov_matrix, left_cluster)
        var_right = get_cluster_variance(cov_matrix, right_cluster)
        alpha = 1.0 - var_left / (var_left + var_right)
        weights[left_cluster] *= alpha
        weights[right_cluster] *= (1.0 - alpha)
        clusters.extend([left_cluster, right_cluster])
    return weights.sort_index()

# --- 3a. Estimate Expected Returns and Covariance (In-Sample) ---
annual_mu_sample = (returns_in_sample.mean().values * 12) - expense_vector
sample_cov_in_sample = returns_in_sample.cov().values
annual_cov_sample = sample_cov_in_sample * 12
lw = LedoitWolf().fit(returns_in_sample.values)
annual_cov_shrunk = lw.covariance_ * 12
annual_mu = annual_mu_sample
n_obs, n_assets = returns_in_sample.shape

# --- 3d. Define In-Sample Benchmark and Select Portfolios ---
voo_mu_annual_in_sample = returns_in_sample['VOO'].mean() * 12 - etf_expense_map.get('VOO', 0.0)
voo_sigma_annual_in_sample = returns_in_sample['VOO'].std() * np.sqrt(12)

ef_shrunk = efficient_frontier(annual_cov_shrunk, annual_mu, n_points=FRONTIER_POINTS)
_, w_sigma_shrunk = select_portfolio(ef_shrunk, 'sigma', voo_sigma_annual_in_sample)

# --- 4a. Resampled Efficient Frontier (In-Sample) ---
print("Running Resampled Frontier on In-Sample data...")
resampled_weights_list = []
for i in range(RESAMPLE_ITERATIONS):
    boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)
    returns_boot = returns_in_sample.iloc[boot_indices]
    mu_boot = (returns_boot.mean().values * 12) - expense_vector
    try:
        cov_boot = LedoitWolf().fit(returns_boot.values).covariance_ * 12
        ef_boot = efficient_frontier(cov_boot, mu_boot, n_points=30)
        _, w_boot = select_portfolio(ef_boot, 'sigma', voo_sigma_annual_in_sample)
        if w_boot is not None: resampled_weights_list.append(w_boot)
    except (ValueError, np.linalg.LinAlgError): continue
w_resampled = np.mean(resampled_weights_list, axis=0) if resampled_weights_list else None

# --- 4c. Black-Litterman Optimization (In-Sample) ---
print("Running Black-Litterman on In-Sample data...")
w_bl_opt = None
try:
    vea_idx, vwo_idx = etf_symbols.index('VEA'), etf_symbols.index('VWO')
    pi = 0.2 * annual_mu_sample + 0.8 * annual_mu_sample.mean()
    P = np.zeros((1, n_assets)); P[0, vea_idx] = 1; P[0, vwo_idx] = -1
    Q = np.array([0.01]); omega = np.diag([0.0025]); bl_tau = 0.05
    cov_inv = np.linalg.inv(bl_tau * annual_cov_shrunk); omega_inv = np.linalg.inv(omega)
    M_inv = np.linalg.inv(cov_inv + P.T @ omega_inv @ P)
    m_bl = M_inv @ (cov_inv @ pi + P.T @ omega_inv @ Q)
    w_bl = cp.Variable(n_assets)
    objective = cp.Maximize(m_bl @ w_bl - 0.5 * cp.quad_form(w_bl, annual_cov_shrunk))
    constraints = [cp.sum(w_bl) == 1, w_bl >= 0]
    problem = cp.Problem(objective, constraints); problem.solve()
    w_bl_opt = w_bl.value
except (ValueError, IndexError): print("Could not perform Black-Litterman.")

# --- 4d. Risk Parity Optimization (In-Sample) ---
print("Running Risk Parity on In-Sample data...")
result = minimize(fun=risk_parity_objective, x0=np.ones(n_assets)/n_assets, args=(annual_cov_sample,),
                  method='SLSQP', bounds=tuple((0.0, 1.0) for _ in range(n_assets)),
                  constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}))
rp_weights = result.x if result.success else None

# --- 4e. Hierarchical Risk Parity (HRP) (In-Sample) ---
print("Running HRP on In-Sample data...")
corr_matrix = returns_in_sample.corr()
dist_matrix = correlation_to_distance(corr_matrix)
linkage_matrix = linkage(squareform(dist_matrix), method='single')
sorted_indices = leaves_list(linkage_matrix)
sorted_tickers = [etf_symbols[i] for i in sorted_indices]
sorted_cov = returns_in_sample[sorted_tickers].cov().values * 12
hrp_weights_sorted = recursive_bisection(sorted_cov, np.arange(len(sorted_tickers)))
hrp_weights = np.zeros(n_assets)
for i, ticker in enumerate(sorted_tickers):
    hrp_weights[etf_symbols.index(ticker)] = hrp_weights_sorted.iloc[i]

# --- 4f. DCC-GARCH Portfolio Optimization ---
# OOS NOTE: The DCC-GARCH model is inherently rolling. We run it over the
# full period to generate a continuous series of weights. Later, we will slice
# its returns to evaluate only the OOS portion.
print("Running DCC-GARCH analysis over full period...")
dynamic_returns_series_dcc = pd.Series(dtype=float)
try:
    price_weekly = yf.download(etf_symbols, start=returns_monthly.index.min(), interval='1wk', auto_adjust=True)['Adj Close'].dropna()
    returns_weekly = np.log(price_weekly / price_weekly.shift(1)).dropna()
    month_ends = returns_weekly.resample('M').last().index
    voo_sigma_annual_weekly = returns_weekly['VOO'].std() * np.sqrt(52)
    rolling_dcc_weights_list = []
    for date in month_ends:
        if len(returns_weekly.loc[:date]) < 104: continue
        try:
            data_window = returns_weekly.loc[:date]
            garch_models = [ConstantMean(data_window[s], GARCH(1,1)).fit(disp='off') for s in etf_symbols]
            dcc_res = DCC(garch_models).fit(disp='off')
            forecasted_cov = dcc_res.forecast(horizon=1).cov.iloc[-1].values * 52
            forecasted_mu = data_window.mean().values * 52
            ef_dcc = efficient_frontier(forecasted_cov, forecasted_mu, n_points=30)
            _, w_dcc = select_portfolio(ef_dcc, 'sigma', voo_sigma_annual_weekly)
            if w_dcc is not None:
                rolling_dcc_weights_list.append(pd.Series(w_dcc, index=etf_symbols, name=date))
        except Exception: continue
    if rolling_dcc_weights_list:
        dcc_weights_df = pd.concat(rolling_dcc_weights_list, axis=1).T
        dcc_shifted_weights = dcc_weights_df.shift(1).reindex(returns_monthly.index).ffill()
        common_idx = dcc_shifted_weights.dropna().index.intersection(returns_monthly.index)
        dynamic_returns_series_dcc = pd.Series(
            np.sum(dcc_shifted_weights.loc[common_idx].values * returns_monthly.loc[common_idx, etf_symbols].values, axis=1),
            index=common_idx
        )
except Exception as e: print(f"DCC-GARCH analysis failed: {e}")


# ==============================================================================
# SECTIONS 5 & 6: REGIME MODEL FITTING (IN-SAMPLE) & DYNAMIC BACKTEST (OOS)
# ==============================================================================
print("\n--- Sections 5 & 6: Fitting Regime Model (In-Sample) & Backtesting (Out-of-Sample) ---")

# --- 5a. Load and Prepare Exogenous Data ---
def get_fred_data(start, end):
    symbols = {'3M': "DGS3MO", '10Y': "DGS10", 'VIX': "VIXCLS"}
    try:
        df = DataReader(list(symbols.values()), 'fred', start, end)
        df = df.rename(columns={v: k for k, v in symbols.items()})
        df[['3M', '10Y']] /= 100.0
        df['Spread_10Y_3M'] = df['10Y'] - df['3M']
        return df.dropna()
    except Exception as e:
        print(f"Could not fetch FRED data: {e}")
        return pd.DataFrame()

exog_df = get_fred_data(returns_monthly.index.min(), returns_monthly.index.max())
exog_monthly = exog_df.resample('M').last().ffill()
common_index = returns_monthly.index.intersection(exog_monthly.index)
exog_lagged = exog_monthly.loc[common_index].shift(1).dropna()
final_index = returns_monthly.index.intersection(exog_lagged.index)
returns_final = returns_monthly.loc[final_index]
exog_final_lagged = exog_lagged.loc[final_index, ['VIX', 'Spread_10Y_3M']]
endog_voo = returns_final['VOO']

# OOS NOTE: Split the final aligned data for regime modeling
final_in_sample_idx = returns_final.index[returns_final.index < oos_start_date]
final_out_of_sample_idx = returns_final.index[returns_final.index >= oos_start_date]

# --- 5c. Fit the Markov Regime-Switching Model (In-Sample) ---
models = {}
for k in range(2, MAX_REGIMES_TO_TEST + 1):
    try:
        mod = MarkovRegression(
            endog=endog_voo.loc[final_in_sample_idx], # Train on in-sample data
            k_regimes=k, trend='c', switching_variance=True,
            exog_tvtp=sm.add_constant(exog_final_lagged.loc[final_in_sample_idx])
        )
        res = mod.fit(search_reps=20)
        if (res.smoothed_marginal_probabilities.idxmax(axis=1).value_counts() < MIN_OBS_PER_REGIME).any(): continue
        models[k] = res
    except Exception: continue
if not models: raise RuntimeError("No suitable regime models could be fitted.")
best_k = min(models, key=lambda k: models[k].bic)
best_model_results = models[best_k]
print(f"Best model selected: {best_k} regimes (trained on in-sample data).")

# --- 6a. Calculate Regime-Specific Optimal Portfolios (In-Sample) ---
full_period_probs = best_model_results.predict(params=best_model_results.params, exog=sm.add_constant(exog_final_lagged))
regime_series_in_sample = full_period_probs.loc[final_in_sample_idx].idxmax(axis=1)

regime_optimal_weights = {}
for i in range(best_k):
    in_regime_periods = regime_series_in_sample[regime_series_in_sample == i].index
    if len(in_regime_periods) < max(24, n_assets):
        regime_optimal_weights[i] = np.array([1.0 if s == 'VOO' else 0.0 for s in etf_symbols])
        continue
    returns_regime = returns_final.loc[in_regime_periods]
    mu_regime = (returns_regime.mean().values * 12) - expense_vector
    cov_regime = LedoitWolf().fit(returns_regime.values).covariance_ * 12
    ef_regime = efficient_frontier(cov_regime, mu_regime, n_points=FRONTIER_POINTS)
    _, w_opt = select_portfolio(ef_regime, 'sigma', voo_sigma_annual_in_sample)
    regime_optimal_weights[i] = w_opt if w_opt is not None else np.array([1.0 if s == 'VOO' else 0.0 for s in etf_symbols])

# --- 6b. OOS NOTE: Backtest the Dynamic Strategy on the Out-of-Sample Period ---
oos_probs = full_period_probs.loc[final_out_of_sample_idx]
dynamic_weights_list_oos = []
for t in range(len(oos_probs)):
    probs_t = oos_probs.iloc[t]
    blended_w = np.zeros(n_assets)
    for i in range(best_k):
        if i in regime_optimal_weights:
            blended_w += probs_t[i] * regime_optimal_weights[i]
    dynamic_weights_list_oos.append(blended_w / blended_w.sum())

dynamic_returns_series_oos = pd.Series(
    np.sum(np.array(dynamic_weights_list_oos) * returns_final.loc[final_out_of_sample_idx, etf_symbols].values, axis=1),
    index=final_out_of_sample_idx
)

# ==============================================================================
# SECTION 7: FINAL **OUT-OF-SAMPLE** PERFORMANCE COMPARISON
# ==============================================================================
print("\n--- Section 7: Comparing All Strategies on OUT-OF-SAMPLE Data ---")

# --- 7a. Define a Performance Metrics Calculator ---
def calculate_performance_metrics(returns_series):
    if returns_series.empty or returns_series.isnull().all():
        return {"Total Return (%)": np.nan, "Annualized Return (%)": np.nan,
                "Annualized Volatility (%)": np.nan, "Sharpe Ratio": np.nan, "Max Drawdown (%)": np.nan}
    n_periods_per_year = 12
    ann_return = ((1 + returns_series.mean()) ** n_periods_per_year - 1) * 100
    ann_vol = returns_series.std() * np.sqrt(n_periods_per_year) * 100
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan
    cumulative_returns = (1 + returns_series).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak - 1) * 100
    max_drawdown = drawdown.min()
    return {"Total Return (%)": total_return, "Annualized Return (%)": ann_return,
            "Annualized Volatility (%)": ann_vol, "Sharpe Ratio": sharpe, "Max Drawdown (%)": max_drawdown}

# --- 7b. OOS NOTE: Prepare All Strategy Returns for OOS Comparison ---
strategies_oos = {
    'VOO Benchmark': returns_out_of_sample['VOO'],
    'Static Shrunk (Risk-Match)': (returns_out_of_sample[etf_symbols] @ w_sigma_shrunk) if w_sigma_shrunk is not None else pd.Series(dtype=float),
    'Static Resampled': (returns_out_of_sample[etf_symbols] @ w_resampled) if w_resampled is not None else pd.Series(dtype=float),
    'Black-Litterman': (returns_out_of_sample[etf_symbols] @ w_bl_opt) if w_bl_opt is not None else pd.Series(dtype=float),
    'Risk Parity': (returns_out_of_sample[etf_symbols] @ rp_weights) if rp_weights is not None else pd.Series(dtype=float),
    'Hierarchical Risk Parity': (returns_out_of_sample[etf_symbols] @ hrp_weights) if hrp_weights is not None else pd.Series(dtype=float),
    'DCC-GARCH Dynamic': dynamic_returns_series_dcc.loc[returns_out_of_sample.index],
    'Regime-Aware Dynamic': dynamic_returns_series_oos
}

# --- 7c. Plot Cumulative OOS Performance ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))
for name, returns in strategies_oos.items():
    if not returns.dropna().empty:
        (1 + returns).cumprod().plot(ax=ax, label=name, lw=2)
ax.set_title('Out-of-Sample Cumulative Performance Comparison', fontsize=16)
ax.set_xlabel('Date', fontsize=12); ax.set_ylabel('Growth of $1 (Log Scale)', fontsize=12)
ax.set_yscale('log'); ax.legend(loc='upper left', fontsize=10)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout(); plt.show()

# --- 7d. Display Final OOS Performance Metrics Table ---
all_perf_metrics_oos = {name: calculate_performance_metrics(ret.dropna()) for name, ret in strategies_oos.items()}
all_perf_df_oos = pd.DataFrame(all_perf_metrics_oos).T

print("\n" + "="*70)
print("      OUT-OF-SAMPLE STRATEGY PERFORMANCE METRICS")
print("="*70)
print(all_perf_df_oos.sort_values(by='Sharpe Ratio', ascending=False))
print("="*70 + "\n")

print("--- Analysis Complete ---")
