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



################################################################ - Updated code 

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

from sklearn.covariance import LedoitWolf
from pandas_datareader.data import DataReader
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# --- Global Settings & Constants ---

# NOTE: You may need to change this to the directory where your script and data file are located.
DIRECTORY = '.'
# The file 'vanguard_etf_list.xlsx' is expected to be in the above directory.
# It should contain ETF metadata like Symbol, Fund name, and Expense ratio.

# Analysis Period
ANALYSIS_YEARS = 10

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
# 1. Drop any ETF (column) that doesn't have at least 80% of the data points in our window.
min_observations = int(len(returns_monthly) * 0.80)
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
    cov_roll = window_data.cov().values * 12

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
    'Dynamic Regime Strategy': dynamic_returns_series
}

# --- 7c. Plot Cumulative Performance of All Strategies ---
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

# --- 7d. Display Final Performance Metrics Table ---
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


