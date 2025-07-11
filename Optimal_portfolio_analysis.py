# ==============================================================================
# SECTION 0: ORIGINAL CODE (with minor consistency fixes)
# ==============================================================================
import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import cvxopt as opt
from pandas_datareader.data import DataReader
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# 1. Settings
# NOTE: You may need to change this directory
DIRECTORY = '.'
os.chdir(DIRECTORY)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 2. Load ETF metadata (assuming 'vanguard_etf_list.xlsx' is in the directory)
try:
    etf_lookup_df = pd.read_excel("vanguard_etf_list.xlsx", skiprows=4)
    etf_name_map = dict(zip(etf_lookup_df['Symbol'], etf_lookup_df['Fund name']))
    etf_expense_map = dict(zip(etf_lookup_df['Symbol'], etf_lookup_df['Expense ratio']))
    etf_name_map = {k: v for k,v in etf_name_map.items() if pd.notna(k)}
    etf_expense_map = {k: v for k,v in etf_expense_map.items() if pd.notna(k)}
    etf_symbols = list(etf_name_map.keys())
except FileNotFoundError:
    print("Warning: 'vanguard_etf_list.xlsx' not found. Using a smaller, predefined list of ETFs.")
    etf_symbols = ['VOO', 'VTI', 'VEA', 'VWO', 'BND', 'BNDX', 'VGIT', 'VGLT', 'VTIP', 'MUB']
    etf_name_map = {s: s for s in etf_symbols} # placeholder
    etf_expense_map = {s: 0.03/100 for s in etf_symbols} # placeholder

# Manual filter
industry_keywords = ['Energy', 'Health Care', 'Consumer', 'Materials', 'Financials', 'Utilities', 'Real Estate', 'Industrials', 'Communication', 'Information Technology']
remove_duplicates = ['VGT', 'VHT', 'VPU', 'VDC', 'VAW', 'VIS', 'VFH', 'VNQ', 'VOX', 'VDE', 'VCR']
def is_industry_or_duplicate(sym):
    name = etf_name_map.get(sym, '')
    return any(kw in name for kw in industry_keywords) or sym in remove_duplicates
etf_symbols = [sym for sym in etf_symbols if not is_industry_or_duplicate(sym)]
etf_symbols = list(dict.fromkeys(etf_symbols)) # remove duplicates from list itself

# 3. Fetch data
def reinvest_dividends(price, div):
    return price * (1 + div / price)

def get_total_return_series(ticker):
    print(f"Processing {ticker}...")
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="max", auto_adjust=False, back_adjust=True)[['Close']].rename(columns={'Close': ticker})
        return df
    except Exception as e:
        print(f"Could not fetch {ticker}: {e}")
        return pd.DataFrame()

# Build DataFrame
prices = pd.DataFrame()
for ticker in etf_symbols:
    tr_df = get_total_return_series(ticker)
    if not tr_df.empty:
        prices = pd.concat([prices, tr_df], axis=1)

prices.index = pd.to_datetime(prices.index).tz_localize(None)
monthly = prices.resample('M').last().pct_change()

# 10-year window
cutoff = monthly.index.max() - pd.DateOffset(years=10)
monthly = monthly[monthly.index >= cutoff]
monthly = monthly.dropna(axis=1, thresh=len(monthly)*0.8) # Keep ETFs with at least 80% of data
monthly = monthly.dropna(axis=0) # Drop any rows with remaining NaNs

etf_symbols = monthly.columns.tolist()
if 'VOO' not in etf_symbols:
    raise ValueError("VOO data is missing or was dropped. It is required as the benchmark.")

expense_vector = np.array([etf_expense_map.get(sym, 0.0) for sym in etf_symbols])

# 4. Compute sample moments and shrinkage
annual_mu = monthly.mean().values * 12 - expense_vector
sample_mu = annual_mu / 12 # monthly mu for simulation


# covariance estimates
sample_cov = monthly.cov().values
annual_cov = sample_cov * 12

lw = LedoitWolf().fit(monthly.values)
annual_cov_shrunk = lw.covariance_ * 12


def james_stein_shrinkage(mu):
    mu_bar = mu.mean()
    shrinkage_factor = 1 - ((len(mu) - 3) * mu.var()) / ((len(mu) - 1) * ((mu - mu_bar) ** 2).sum())
    shrinkage_factor = max(0, min(shrinkage_factor, 1))
    return shrinkage_factor * mu_bar + (1 - shrinkage_factor) * mu


annual_mu_shrunk = james_stein_shrinkage(annual_mu)

print("\nSample mu (annual):", annual_mu.round(3))
print("Shrinked mu (annual):", annual_mu_shrunk.round(3))
#=> very aggressive shrinkage.  This is not useful and we should ignore this estimator

# 5. Optimizer
opt.solvers.options['show_progress'] = False
def efficient_frontier(cov_mat, mu_vec, n_points=50):
    n = len(mu_vec)
    P = opt.matrix(cov_mat)
    q = opt.matrix(np.zeros((n,1)))
    A = opt.matrix(np.vstack([mu_vec, np.ones((1,n))]))
    G = opt.matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = opt.matrix(np.vstack([np.zeros((n,1)), np.ones((n,1))]))
    mus = np.linspace(mu_vec.min(), mu_vec.max(), n_points)
    frontier = {'mu': [], 'sigma': [], 'weights': []}
    for mu in mus:
        b = opt.matrix([mu, 1.0])
        try:
            sol = opt.solvers.qp(P, q, G, h, A, b)
            if sol['status'] == 'optimal':
                w = np.array(sol['x']).flatten()
                sigma = np.sqrt(w @ cov_mat @ w)
                frontier['mu'].append(mu)
                frontier['sigma'].append(sigma)
                frontier['weights'].append(w)
        except ValueError:
            pass # Solver failed for this target return
    return frontier

# 6. Benchmark metrics
voo_monthly = monthly['VOO']
voo_mu_ann = voo_monthly.mean() * 12 - etf_expense_map.get('VOO', 0.0)
voo_sigma_ann = voo_monthly.std() * np.sqrt(12)

def select_by_mu(frontier, target_mu):
    if not frontier['mu']: return None, None
    diffs = np.abs(np.array(frontier['mu']) - target_mu)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]

def select_by_sigma(frontier, target_sigma):
    if not frontier['sigma']: return None, None
    diffs = np.abs(np.array(frontier['sigma']) - target_sigma)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]


# 8. Generate frontiers
ef_raw = efficient_frontier(annual_cov, annual_mu)
ef_shrunk = efficient_frontier(annual_cov_shrunk, annual_mu)

idx_mu, w_mu_raw = select_by_mu(ef_raw, voo_mu_ann)
idx_sigma, w_sigma_raw = select_by_sigma(ef_raw, voo_sigma_ann)

idx_mu_shrunk, w_mu_shrunk = select_by_mu(ef_shrunk, voo_mu_ann)
idx_sigma_shrunk, w_sigma_shrunk = select_by_sigma(ef_shrunk, voo_sigma_ann)


# Show top-3 ETFs by weight for each selected portfolio
for label, w in [('Mu-matched', w_mu_raw), 
                 ('Sigma-matched', w_sigma_raw),
                 ('Mu-matched - shrinkage', w_mu_shrunk), 
                 ('Sigma-matched - shrinkage', w_sigma_shrunk)]:
    top_idx = np.argsort(w)[-3:][::-1]
    print(f"\nTop 3 ETFs for {label} portfolio:")
    for i in top_idx:
        print(f"  {etf_symbols[i]} ({etf_name_map.get(etf_symbols[i], 'Unknown')}): {w[i]:.3%}")

# Plotting historical performance
def cumulative_returns(weights):
    port_returns = monthly[etf_symbols].values @ weights
    port_returns = pd.Series(port_returns, index=monthly.index)
    return (1 + port_returns).cumprod()

cum_voo = (1 + voo_monthly).cumprod()
cum_mu = cumulative_returns(w_mu_raw)
cum_sigma = cumulative_returns(w_sigma_raw)

plt.figure(figsize=(10,6))
plt.plot(cum_voo, label='VOO')
plt.plot(cum_mu, label='Mu-matched Portfolio')
plt.plot(cum_sigma, label='Sigma-matched Portfolio')
plt.title('Historical Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 9. Monte Carlo simulation
n_scenarios = 10000
horizon = 120  # months
rng = np.random.default_rng(0)
sim = rng.multivariate_normal(sample_mu, sample_cov, size=(n_scenarios, horizon))

def simulate_performance(weights):
    # simulate portfolio returns
    port_r = sim @ weights
    # net of expense drag
    drag = weights @ expense_vector
    return {
        'mean': np.mean(port_r)*12 - drag,
        'vol': np.std(port_r)*np.sqrt(12),
        'VaR_5': np.percentile(port_r,5)*12
    }

perf_mu = {
    'Raw': simulate_performance(w_mu_raw),
    'Shrunk': simulate_performance(w_mu_shrunk),
    'VOO': {'mean': voo_mu_ann, 'vol': voo_sigma_ann}
}

perf_sigma = {
    'Raw': simulate_performance(w_sigma_raw),
    'Shrunk': simulate_performance(w_sigma_shrunk),
    'VOO': {'mean': voo_mu_ann, 'vol': voo_sigma_ann}
}

#  Display results
print("\nPerformance Summary - Mu matched to VOO:")
print(pd.DataFrame(perf_mu).T)

print("\nPerformance Summary - Sigma matched to VOO:")
print(pd.DataFrame(perf_sigma).T)

# Shrinkage is not useful, if matching on variance, it select a negative return portfolio 

# Test different optimization methods
# --- Resampled Efficient Frontier ---
B = 100
n_obs = monthly.shape[0]
resampled_weights = {
    'Raw': [],
    'Shrunk': []
}

for label in ['Raw', 'Shrunk']:
    for b in range(B):
        if (b + 1) % 50 == 0 or b == 0:
            print(f"Resample simulation {b+1}/{B} ({label})")

        idxs = np.random.choice(n_obs, n_obs, replace=True)
        boot = monthly.values[idxs, :]

        boot_df = pd.DataFrame(boot, columns=etf_symbols)
        boot_clean = boot_df.dropna()

        if boot_clean.shape[0] < 2:
            continue  # not enough observations

        mu_b = boot_clean.mean().values * 12 - expense_vector

        if label == 'Shrunk':
            try:
                lw_b = LedoitWolf().fit(boot_clean.values)
                cov_b = lw_b.covariance_ * 12
            except ValueError:
                continue  # skip if shrinkage fails
        else:
            cov_b = np.cov(boot_clean.values, rowvar=False) * 12

        # Skip if NaNs or bad numbers
        if (
            np.isnan(mu_b).any() or np.isnan(cov_b).any()
            or np.isinf(mu_b).any() or np.isinf(cov_b).any()
        ):
            continue

        try:
            ef_b = efficient_frontier(cov_b, mu_b)
            _, w_b = select_by_sigma(ef_b, voo_sigma_ann)
            resampled_weights[label].append(w_b)
        except (ValueError, np.linalg.LinAlgError):
            continue


avg_w_resampled = {
    k: np.mean(v, axis=0) for k, v in resampled_weights.items()
}

for label, w in avg_w_resampled.items():
    top_idx = np.argsort(w)[-5:][::-1]
    print(f"\nTop 5 ETFs for Resampled {label} Portfolio:")
    for i in top_idx:
        print(f"  {etf_symbols[i]} ({etf_name_map.get(etf_symbols[i], 'Unknown')}): {w[i]:.3%}")




# --- Rolling Window Estimation ---
# Track changes in portfolio weights over time using rolling window optimization

window_size = 60  # months
step = 1  # step forward each month
rolling_dates = monthly.index[window_size::step]
rolling_weights = []

for end_date in rolling_dates:
    data_window = monthly.loc[:end_date].iloc[-window_size:]
    mu_roll = data_window.mean().values * 12 - expense_vector
    cov_roll = data_window.cov().values * 12  # sample covariance avoids NaNs
    ef_roll = efficient_frontier(cov_roll, mu_roll)
    _, w_roll = select_by_sigma(ef_roll, voo_sigma_ann)
    rolling_weights.append(pd.Series(w_roll, index=etf_symbols, name=end_date))

# Combine into DataFrame for visualization or export
rolling_weights_df = pd.concat(rolling_weights, axis=1).T

# Plot top 3 weight trajectories
top_etfs = rolling_weights_df.mean().sort_values(ascending=False).head(5).index
rolling_weights_df[top_etfs].plot(figsize=(10,6), title='Top ETF Weights Over Time (Rolling Optimization)')
plt.ylabel("Weight")
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Plot frontiers and VOO benchmark
plt.figure(figsize=(8,6))
plt.plot(ef_raw['sigma'], ef_raw['mu'], label='Raw Frontier')
plt.plot(ef_shrunk['sigma'], ef_shrunk['mu'], label='Shrunk Frontier')
#plt.scatter([voo_sigma], [voo_mu], color='k', label='VOO')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
# SECTION 1: LOAD EXOGENOUS DATA (YIELD CURVE & VIX)
# ==============================================================================
print("\n--- Section 1: Loading Exogenous Data for Regime Modeling ---")


start_date = monthly.index.min()
end_date = monthly.index.max()

# 1a. Fetch Yield Curve Data
def get_yield_curve(start, end):
    symbols = {
        '1M': "DGS1MO",
        '3M': "DGS3MO",
        '6M': "DGS6MO",
        '1Y': "DGS1",
        '2Y': "DGS2",
        '5Y': "DGS5",
        '10Y': "DGS10",
        '30Y': "DGS30"
    }

    df = pd.DataFrame()
    for label, fred_code in symbols.items():
        try:
            data = DataReader(fred_code, 'fred', start, end)
            df[label] = data[fred_code]
        except Exception as e:
            print(f"Error fetching {label}: {e}")
            df[label] = np.nan
    df = df / 100
    df['Spread'] = df['10Y'] - df['3M']
    return df

yield_curve_df = get_yield_curve(start_date, end_date)
yield_curve_df.index = yield_curve_df.index.tz_localize(None)

def plot_risk_free_and_spread(short_term_rate, long_term_rate):
    """
    Plots the risk-free rate (3M) and the 10Y-3M yield spread over time.
    """
    yield_spread = long_term_rate - short_term_rate

    plt.figure(figsize=(10, 5))
    plt.plot(short_term_rate.index, short_term_rate * 100, label='3M Rate (Risk-Free)', color='blue')
    plt.plot(yield_spread.index, yield_spread * 100, label='10Y - 3M Spread', color='green')
    plt.title("Risk-Free Rate and Yield Spread Over Time")
    plt.ylabel("Percent (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# You can now use the yield curve to create regimes (e.g., low vs high short rates or yield curve slope)
short_term_rate = yield_curve_df['3M']
long_term_rate = yield_curve_df['10Y']
yield_spread = long_term_rate - short_term_rate


# Plot risk-free rate and yield spread over time
plot_risk_free_and_spread(short_term_rate, long_term_rate)
# -> there appear to be four regimes, let's model those


# 1b. Fetch VIX Data
print("Fetching VIX data...")
vix = yf.Ticker('^VIX').history(start=start_date, end=end_date)[['Close']]
vix.index = pd.to_datetime(vix.index).tz_localize(None)
vix.rename(columns={'Close': 'VIX'}, inplace=True)

# 1c. Combine and Align Data
exog_df = pd.concat([yield_curve_df, vix], axis=1)
exog_df = exog_df.resample('M').last().ffill().dropna()

# Align all data to a common index
common_index = monthly.index.intersection(exog_df.index)
monthly_aligned = monthly.loc[common_index]
exog_aligned = exog_df.loc[common_index]

# Create lagged exogenous variables to predict NEXT period's regime
exog_lagged = exog_aligned.shift(1).dropna()

# Final alignment after lagging
final_index = monthly_aligned.index.intersection(exog_lagged.index)
monthly_final = monthly_aligned.loc[final_index]
exog_final_lagged = exog_lagged.loc[final_index]

# We will model the regimes of the broad market, using VOO as the endogenous variable
endog = monthly_final['VOO']

print(f"Final dataset for modeling has {len(endog)} monthly observations.")


# ==============================================================================
# SECTION 2: MARKOV REGIME-SWITCHING MODEL ESTIMATION (WITH SWITCHING VARIANCE)
# ==============================================================================
print("\n--- Section 2: Estimating Markov Regime-Switching Models ---")
print("NOTE: Now allowing both Mean and Variance to switch between regimes.")

MIN_OBSERVATIONS = 10  # set your minimum number of observations per regime here

models = {}
for k in range(2, 5):
    print(f"Fitting model with {k} regimes...")
    try:
        mod = MarkovRegression(
            endog,
            k_regimes=k,
            trend='c',
            switching_variance=True,  # Allows volatility to be regime-specific
            exog_tvtp=sm.add_constant(exog_final_lagged)
        )
        res = mod.fit(search_reps=20)
        
        # Get regime assignment by highest smoothed probability
        smoothed_probs = res.smoothed_marginal_probabilities
        assigned_regimes = smoothed_probs.idxmax(axis=1)
        
        # Count observations per regime
        counts = assigned_regimes.value_counts()
        
        # Check if all regimes meet minimum observation count
        if (counts < MIN_OBSERVATIONS).any():
            print(f"  > Model with {k} regimes rejected: regime with insufficient observations")
            continue
        
        models[k] = res
        print(f"  > BIC: {res.bic:.2f}, Log-Likelihood: {res.llf:.2f}")
        
    except Exception as e:
        print(f"  > Failed to fit model with {k} regimes: {e}")

if not models:
    raise RuntimeError("No suitable models found with all regimes having minimum observations")

best_k = min({k: v.bic for k, v in models.items()})
best_res = models[best_k]

print(f"\nBest model selected: {best_k} regimes (BIC = {best_res.bic:.2f})")
print("\nModel Parameters (Sorted by Volatility):")

# Sort regimes by volatility (sigma2) for consistent labeling
regime_vols = best_res.params.filter(like='sigma2').sort_values()
regime_order = regime_vols.index.str.extract(r'\[(\d+)\]')[0].astype(int)
regime_map = {old: new for new, old in enumerate(regime_order)}

# Extract and display sorted parameters
sorted_params = pd.DataFrame()
for i in range(best_k):
    original_regime_idx = regime_order.iloc[i]
    mean_param = f'const[{original_regime_idx}]'
    vol_param = f'sigma2[{original_regime_idx}]'
    
    ann_mean = best_res.params[mean_param] * 12 * 100 # In %
    ann_vol = np.sqrt(best_res.params[vol_param]) * np.sqrt(12) * 100 # In %
    
    sorted_params[f'Regime {i}'] = [f'{ann_mean:.1f}%', f'{ann_vol:.1f}%']

sorted_params.index = ['Annualized Mean (VOO)', 'Annualized Volatility (VOO)']
print(sorted_params)


# Apply the new sorted order to probabilities and regime series
smoothed_probs = best_res.smoothed_marginal_probabilities.rename(columns=regime_map).sort_index(axis=1)
regime_series = smoothed_probs.idxmax(axis=1).rename('regime')


# ==============================================================================
# SECTION 3: REGIME-SPECIFIC EFFICIENT FRONTIERS 
# ==============================================================================
print("\n--- Section 3: Analyzing Regime-Specific Efficient Frontiers ---")

regime_frontiers = {}
regime_optimal_weights = {}

plt.figure(figsize=(12, 7))
colors = plt.cm.plasma(np.linspace(0, 1, best_k))

for i in range(best_k):
    print(f"\nAnalyzing Regime {i}...")
    in_regime = (regime_series == i)
    min_obs = max(6, len(etf_symbols) // 2)  # at least 6 months or half the number of assets
    if in_regime.sum() < min_obs:
        print(f"  > Skipping Regime {i}, not enough data points ({in_regime.sum()})")
        continue

    regime_monthly = monthly_final[in_regime]
    
    # 1. Estimate sample moments
    mu_regime_sample = regime_monthly.mean().values * 12 - expense_vector
    lw_regime = LedoitWolf().fit(regime_monthly.values)
    cov_regime = lw_regime.covariance_ * 12
    
    # 2. Apply mean shrinkage
    # mu_regime_shrunk = james_stein_shrinkage(mu_regime_sample)
    # -> let us not do this, the shrinkage is every aggressive

    mu_regime = mu_regime_sample

    # 3. Generate efficient frontier with shrunk means
    ef_regime = efficient_frontier(cov_regime, mu_regime)
    regime_frontiers[i] = ef_regime

    _, w_opt = select_by_sigma(ef_regime, voo_sigma_ann)
    if w_opt is not None:
        regime_optimal_weights[i] = w_opt
        print(f"  > Optimal portfolio for Regime {i} (matching VOO vol):")
        top_idx = np.argsort(w_opt)[-3:][::-1]
        for idx in top_idx:
            if w_opt[idx] > 0.01:
                print(f"    - {etf_symbols[idx]}: {w_opt[idx]:.1%}")
    else:
        print("  > Could not find an optimal portfolio for this regime.")
        regime_optimal_weights[i] = np.array([1.0 if s == 'VOO' else 0.0 for s in etf_symbols]) # Fallback to VOO

    if ef_regime['mu']:
        plt.plot(ef_regime['sigma'], ef_regime['mu'], label=f'Regime {i} Frontier', color=colors[i], lw=2)
        if w_opt is not None:
            opt_sigma = np.sqrt(w_opt @ cov_regime @ w_opt)
            opt_mu = w_opt @ mu_regime # Use shrunk mu for plotting
            plt.scatter(opt_sigma, opt_mu, marker='*', s=200, color=colors[i], zorder=5, edgecolors='black')

plt.scatter([voo_sigma_ann], [voo_mu_ann], color='black', marker='X', s=150, label='VOO (Overall)', zorder=5)
plt.title('Efficient Frontiers for Each Market Regime (with Mean Shrinkage)')
plt.xlabel('Annualized Volatility (Sigma)')
plt.ylabel('Annualized Return (Mu)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================================================================
# SECTION 4: BACKTESTING THE DYNAMIC REGIME-SWITCHING STRATEGY
# ==============================================================================
print("\n--- Section 4: Backtesting the Dynamic Strategy ---")

predicted_probs = best_res.smoothed_marginal_probabilities.copy()
predicted_probs.columns = predicted_probs.columns.map(regime_map)  # Apply sorted regime map
predicted_probs = predicted_probs.sort_index(axis=1)

dynamic_weights = []
for t in range(len(monthly_final)):
    probs_t = predicted_probs.iloc[t]
    blended_w = np.zeros(len(etf_symbols))
    for i in range(best_k):
        if i in regime_optimal_weights:
            blended_w += probs_t[i] * regime_optimal_weights[i]
    blended_w /= blended_w.sum()
    dynamic_weights.append(blended_w)

dynamic_port_returns = np.sum(np.array(dynamic_weights) * monthly_final[etf_symbols].values, axis=1)
dynamic_returns_series = pd.Series(dynamic_port_returns, index=monthly_final.index)

cum_dynamic = (1 + dynamic_returns_series).cumprod()
cum_voo_aligned = (1 + monthly_final['VOO']).cumprod()

# ==============================================================================
# SECTION 5: PERFORMANCE PLOTTING AND ANALYSIS
# ==============================================================================
print("\n--- Section 5: Performance Analysis ---")

# 5a. Plot Cumulative Performance
plt.figure(figsize=(12, 7))
plt.plot(cum_dynamic, label='Dynamic Strategy (Mean Shrinkage + Switching Vol)', color='crimson', lw=2)
plt.plot(cum_voo_aligned, label='VOO Benchmark', color='navy', linestyle='--')
plt.title('Cumulative Performance: Dynamic Strategy vs. VOO Benchmark')
plt.xlabel('Date')
plt.ylabel('Cumulative Growth of $1')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 5b. Performance Metrics
def calculate_metrics(returns_series):
    cum_ret = (1 + returns_series).cumprod()
    total_return = (cum_ret.iloc[-1] - 1) * 100  # ✅ correct
    ann_return = ((1 + returns_series.mean()) ** 12 - 1) * 100
    ann_vol = returns_series.std() * np.sqrt(12) * 100
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret / peak - 1) * 100
    max_drawdown = drawdown.min()
    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": ann_return,
        "Annualized Volatility (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown
    }

performance_df = pd.DataFrame({
    'Dynamic Strategy': calculate_metrics(dynamic_returns_series),
    'VOO Benchmark': calculate_metrics(monthly_final['VOO'])
}).T

print("\nPerformance Metrics Comparison:")
print(performance_df)

# 5c. Plot smoothed probabilities with VOO returns
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
smoothed_probs.plot(ax=axes[0], kind='area', stacked=True, colormap='plasma', alpha=0.8)
axes[0].set_title('Smoothed Probabilities of Each Regime Over Time')
axes[0].set_ylabel('Probability')
axes[0].legend(title='Regime', loc='upper left')

monthly_final['VOO'].plot(ax=axes[1], color='black', label='VOO Monthly Return')
axes[1].set_title('VOO Monthly Returns')
axes[1].set_ylabel('Return')
axes[1].grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



# ==============================================================================
# STRATEGY COMPARISON OF RAW, SHRUNK, BOOTSTRAPPED, AND REGIME-AWARE
# ==============================================================================
print("\n--- Section 6: Strategy Comparison ---")

# Cumulative returns
cum_shrunk = cumulative_returns(w_mu_shrunk)
def safe_cumulative_returns(weight_input, fallback_index):
    if weight_input is not None and len(weight_input) > 0:
        return cumulative_returns(weight_input)
    else:
        return pd.Series(index=fallback_index, data=np.nan)

cum_resampled = safe_cumulative_returns(avg_w_resampled.get('Shrunk'), monthly.index)

# Align for plotting
aligned_index = cum_voo.index.intersection(cum_shrunk.index).intersection(cum_resampled.index)
cum_perf_df = pd.DataFrame({
    'VOO': cum_voo.loc[aligned_index],
    'Raw Portfolio': cum_mu.loc[aligned_index],
    'Shrunk Portfolio': cum_shrunk.loc[aligned_index],
    'Resampled Portfolio': cum_resampled.loc[aligned_index],
    'Dynamic Regime Strategy': cum_dynamic.loc[cum_dynamic.index.intersection(aligned_index)]
})

# Plot all strategies
plt.figure(figsize=(12, 7))
cum_perf_df.plot(ax=plt.gca(), lw=2)
plt.title("Cumulative Returns Comparison Across Strategies")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# ==============================================================================
# CALCULATE PERFORMANCE METRICS
# ==============================================================================
print("\n--- Section 7: Performance Metrics for All Strategies ---")

def safe_weighted_return(weights, fallback_index):
    if weights is not None and len(weights) > 0:
        return (monthly[etf_symbols] @ weights).reindex(fallback_index)
    else:
        return pd.Series(index=fallback_index, data=np.nan)

strategies = {
    'VOO Benchmark': monthly_final['VOO'],
    'Raw Portfolio': (monthly[etf_symbols] @ w_mu_raw).reindex(monthly_final.index),
    'Shrunk Portfolio': (monthly[etf_symbols] @ w_mu_shrunk).reindex(monthly_final.index),
    'Resampled Portfolio': safe_weighted_return(avg_w_resampled.get('Shrunk'), monthly_final.index),
    'Dynamic Regime Strategy': dynamic_returns_series.reindex(monthly_final.index)
}

all_perf_df = pd.DataFrame({
    name: calculate_metrics(returns.dropna())
    for name, returns in strategies.items()
}).T

print("\nComprehensive Strategy Performance Metrics:")
print(all_perf_df)


