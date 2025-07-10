import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import cvxopt as opt
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import invwishart
from pandas_datareader.data import DataReader
import datetime
import ruptures as rpt
from sklearn.preprocessing import StandardScaler



# 1. Settings
DIRECTORY = '/Users/dominikjurek/Library/CloudStorage/Dropbox/Personal/Investment'
os.chdir(DIRECTORY)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 2. Load ETF metadata
etf_lookup_df = pd.read_excel("vanguard_etf_list.xlsx", skiprows=4)
etf_name_map = dict(zip(etf_lookup_df['Symbol'], etf_lookup_df['Fund name']))
etf_expense_map = dict(zip(etf_lookup_df['Symbol'], etf_lookup_df['Expense ratio']))
etf_name_map = {k: v for k,v in etf_name_map.items() if pd.notna(k)}
etf_expense_map = {k: v for k,v in etf_expense_map.items() if pd.notna(k)}
etf_symbols = list(etf_name_map.keys())

# Manual filter: remove industry-specific ETFs and highly overlapping ones
industry_keywords = ['Energy', 'Health Care', 'Consumer', 'Materials', 'Financials', 'Utilities', 'Real Estate', 'Industrials', 'Communication', 'Information Technology']
remove_duplicates = [
    'VGT', 'VHT', 'VPU', 'VDC', 'VAW', 'VIS', 'VFH', 'VNQ', 'VOX', # sector-specific
    'VDE', 'VCR', 'VDC', 'VPU', 'VAW', 'VHT', 'VOX', 'VFH', 'VNQ'  # more sectors
]

def is_industry_or_duplicate(sym):
    name = etf_name_map.get(sym, '')
    return any(kw in name for kw in industry_keywords) or sym in remove_duplicates

etf_symbols = [sym for sym in etf_symbols if not is_industry_or_duplicate(sym)]


# 3. Fetch data
def get_total_return_series(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="max", auto_adjust=True)[['Close']].rename(columns={'Close': ticker})
    tr = df[ticker]
    # reinvest dividends
    divs = t.dividends
    for date, div in divs.items():
        if date in tr.index:
            tr.loc[date:] *= (1 + div / tr.loc[date])
    return tr

# Build DataFrame
prices = pd.DataFrame()
for ticker in etf_symbols:
    print(f"Processing {ticker}...")
    tr = get_total_return_series(ticker)
    prices = pd.concat([prices, tr], axis=1)

prices.index = pd.to_datetime(prices.index) 

monthly = prices.resample('M').last().pct_change()
# 10-year window
cutoff = monthly.index.max() - pd.DateOffset(years=10)
monthly = monthly[monthly.index >= cutoff]
monthly = monthly.dropna(axis=1, thresh=60) # minimum of 5 years of data - remove NAs
etf_symbols = monthly.columns.tolist()

# Align expense vector to the filtered symbols
expense_vector = np.array([etf_expense_map.get(sym, 0.0) for sym in etf_symbols])


# 4. Compute sample moments and shrinkage
# compute gross annualized returns
sample_mu_raw = monthly.mean().values * 12
# subtract annual expenses
annual_mu = sample_mu_raw - expense_vector
# define sample mu after expenses for Monte Carlo simulation of monthly retunrs
sample_mu = annual_mu / 12

# covariance estimates
sample_cov = monthly.cov().values

monthly_clean = monthly.dropna()
lw = LedoitWolf().fit(monthly_clean.values)
cov_shrunk = lw.covariance_

# annualize covariances
annual_cov = sample_cov * 12
annual_cov_shrunk = cov_shrunk * 12

# 5. Optimizer using cvxopt
# Disable cvxopt solver output globally
opt.solvers.options['show_progress'] = False

def efficient_frontier(cov_mat, mu_vec, n_points=50):
    n = len(mu_vec)
    P = opt.matrix(cov_mat)
    q = opt.matrix(np.zeros((n,1)))
    
    # Equality constraints: expected return and full investment
    A = opt.matrix(np.vstack([mu_vec, np.ones((1,n))]))
    
    # constraints
    
    # Inequality constraints: 0 ≤ x ≤ 1
    G = opt.matrix(np.vstack([-np.eye(n), # x_i ≥ 0
                              np.eye(n)])) # x_i ≤ 1
    h = opt.matrix(np.vstack([np.zeros((n,1)), 
                              np.ones((n,1))]))
    
    mus = np.linspace(mu_vec.min(), mu_vec.max(), n_points)
    frontier = {'mu': [], 'sigma': [], 'weights': []}
    for mu in mus:
        b = opt.matrix([mu, 1.0])
        
        sol = opt.solvers.qp(P, q, G, h, A, b)
        w = np.array(sol['x']).flatten()
        sigma = np.sqrt(w @ cov_mat @ w)
        frontier['mu'].append(mu)
        frontier['sigma'].append(sigma)
        frontier['weights'].append(w)
    return frontier

# raw vs shrunk
ef_raw = efficient_frontier(annual_cov, annual_mu)
ef_shrunk = efficient_frontier(annual_cov_shrunk, annual_mu)

# 6. Benchmark metrics - use VOO
voo_monthly = monthly['VOO']
voo_mu = voo_monthly.mean() * 12 - (etf_expense_map.get('VOO', 0.0)/100)
voo_sigma = np.std(voo_monthly) * np.sqrt(12)

# 7. Select portfolio matching VOO's mu

def select_by_mu(frontier, target_mu):
    # find index of frontier with mu closest to target_mu
    diffs = np.abs(np.array(frontier['mu']) - target_mu)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]

idx_raw_mu, w_raw_mu = select_by_mu(ef_raw, voo_mu)
idx_shrunk_mu, w_shrunk_mu = select_by_mu(ef_shrunk, voo_mu)

def select_by_sigma(frontier, target_sigma):
    # find index of frontier with sigma closest to target_sigma
    diffs = np.abs(np.array(frontier['sigma']) - target_sigma)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]

idx_raw_sigma, w_raw_sigma = select_by_sigma(ef_raw, voo_sigma)
idx_shrunk_sigma, w_shrunk_sigma = select_by_sigma(ef_shrunk, voo_sigma)
def select_by_mu(frontier, target_mu):
    diffs = np.abs(np.array(frontier['mu']) - target_mu)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]

def select_by_sigma(frontier, target_sigma):
    diffs = np.abs(np.array(frontier['sigma']) - target_sigma)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]

# 6. Benchmark metrics
voo_monthly = monthly['VOO']
voo_mu = voo_monthly.mean() * 12 - etf_expense_map.get('VOO', 0.0)
voo_sigma = voo_monthly.std() * np.sqrt(12)


# 8. Generate frontiers
ef_raw = efficient_frontier(annual_cov, annual_mu)
ef_shrunk = efficient_frontier(annual_cov_shrunk, annual_mu)

idx_mu, w_mu = select_by_mu(ef_raw, voo_mu)
idx_sigma, w_sigma = select_by_sigma(ef_raw, voo_sigma)

idx_mu_shrunk, w_mu_shrunk = select_by_mu(ef_shrunk, voo_mu)
idx_sigma_shrunk, w_sigma_shrunk = select_by_sigma(ef_shrunk, voo_sigma)


# Show top-3 ETFs by weight for each selected portfolio
for label, w in [('Mu-matched', w_mu), 
                 ('Sigma-matched', w_sigma),
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
cum_mu = cumulative_returns(w_mu)
cum_sigma = cumulative_returns(w_sigma)

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
    'Raw': simulate_performance(w_raw_mu),
    'Shrunk': simulate_performance(w_shrunk_mu),
    'VOO': {'mean': voo_mu, 'vol': voo_sigma}
}

perf_sigma = {
    'Raw': simulate_performance(w_raw_sigma),
    'Shrunk': simulate_performance(w_shrunk_sigma),
    'VOO': {'mean': voo_mu, 'vol': voo_sigma}
}

#  Display results
print("\nPerformance Summary - Mu matched to VOO:")
print(pd.DataFrame(perf_mu).T)

print("\nPerformance Summary - Sigma matched to VOO:")
print(pd.DataFrame(perf_sigma).T)




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
            _, w_b = select_by_mu(ef_b, voo_mu)
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


# --- Hierarchical Risk Parity (HRP) ---
def correl_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

def hrp_weights(cov):
    corr = np.corrcoef(cov)
    dist = correl_dist(corr)
    dist_mat = squareform(dist, checks=False)
    link = linkage(dist_mat, method='single')
    cluster = fcluster(link, t=1.15, criterion='distance')
    sort_ix = np.argsort(cluster)
    cov_ordered = cov[sort_ix][:, sort_ix]
    ivp = 1. / np.diag(cov_ordered)
    ivp /= ivp.sum()
    hrp_w = np.zeros(len(cov))
    hrp_w[sort_ix] = ivp
    return hrp_w

w_hrp = hrp_weights(annual_cov)
top_idx_hrp = np.argsort(w_hrp)[-3:][::-1]
print("\nTop 3 ETFs for HRP Portfolio:")
for i in top_idx_hrp:
    print(f"  {etf_symbols[i]} ({etf_name_map.get(etf_symbols[i], 'Unknown')}): {w_hrp[i]:.3%}")

# --- Bayesian Mean-Variance Optimization ---
def bayesian_mean_variance(mu, cov, tau=0.025):
    n = len(mu)
    prior_mu = np.mean(mu) * np.ones_like(mu)
    cov_prior = cov / tau
    cov_post = np.linalg.inv(np.linalg.inv(cov_prior) + np.linalg.inv(cov))
    mu_post = cov_post @ (np.linalg.inv(cov_prior) @ prior_mu + np.linalg.inv(cov) @ mu)
    inv_cov = np.linalg.inv(cov_post)
    w = inv_cov @ mu_post
    w /= w.sum()
    return w

w_bayes = bayesian_mean_variance(annual_mu, annual_cov)
top_idx_bayes = np.argsort(w_bayes)[-3:][::-1]
print("\nTop 3 ETFs for Bayesian Mean-Variance Portfolio:")
for i in top_idx_bayes:
    print(f"  {etf_symbols[i]} ({etf_name_map.get(etf_symbols[i], 'Unknown')}): {w_bayes[i]:.3%}")



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
    _, w_roll = select_by_mu(ef_roll, voo_mu)
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



#--- Estimate regime change using yield curve and risk-free rate ----
def get_yield_curve(start_date, end_date):
    """
    Fetch yield curve data (1M to 30Y) from FRED using pandas_datareader.
    Returns a DataFrame indexed by date.
    """
    # Normalize dates to be timezone-naive
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)

    
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

    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.to_pydatetime()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.to_pydatetime()

    df = pd.DataFrame()
    for label, fred_code in symbols.items():
        print(f"Fetching yield: {label}")
        try:
            data = DataReader(fred_code, 'fred', start_date, end_date)
            df[label] = data[fred_code]
        except Exception as e:
            print(f"Error fetching {label}: {e}")
            df[label] = np.nan

    return df / 100  # Convert to decimals


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


# Make all datetime indices timezone-naive
monthly.index = monthly.index.tz_localize(None)

# Fetch and plot yield curve
yield_curve_df = get_yield_curve(monthly.index.min(), monthly.index.max())
yield_curve_df.index = yield_curve_df.index.tz_localize(None)


# You can now use the yield curve to create regimes (e.g., low vs high short rates or yield curve slope)
short_term_rate = yield_curve_df['3M']
long_term_rate = yield_curve_df['10Y']
yield_spread = long_term_rate - short_term_rate


# Plot risk-free rate and yield spread over time
plot_risk_free_and_spread(short_term_rate, long_term_rate)
# -> there appear to be four regimes, let's model those

#--- Identify break points ----
# Combine features: short-term rate, long-term rate, and spread
features_df = pd.concat([short_term_rate, long_term_rate, yield_spread], axis=1).dropna()
features_df.columns = ['short', 'long', 'spread']

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

model = rpt.Binseg(model="l2").fit(features_scaled)
change_points = model.predict(n_bkps=3) # 3 breakpoints = 4 regimes

# Build regime series from change points
yield_regime_series = pd.Series(index=features_df.index, dtype=int)
regime_id = 0
start = 0
for cp in change_points:
    yield_regime_series.iloc[start:cp] = regime_id
    regime_id += 1
    start = cp

# Align with monthly_returns
yield_regime_series.index = yield_regime_series.index.tz_localize(None)
aligned_yield_regime_series = yield_regime_series.reindex(
    monthly.index, method='ffill')

# Output regime summary
print("\nRegime counts based on yield curve dynamics:")
print(aligned_yield_regime_series.value_counts(dropna=False))









# 12. Forward-Looking Regime-Based Rebalancing
# Drop rows with any NaNs to ensure GaussianMixture can be fit
monthly_clean = monthly.dropna()

hmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
hmm.fit(monthly_clean.values)
raw_regimes = pd.Series(hmm.predict(monthly_clean.values), index=monthly_clean.index, name='regime')

# Reindex to full index, forward fill to maintain continuity
daily_index = monthly.index
raw_regimes = raw_regimes.reindex(daily_index, method='ffill')

# Filter regimes by requiring a minimum duration
min_duration = 3  # in months
filtered_regimes = raw_regimes.copy()
regime_blocks = (raw_regimes != raw_regimes.shift()).cumsum()
counts = regime_blocks.value_counts().to_dict()
for block_id, count in counts.items():
    if count < min_duration:
        mask = (regime_blocks == block_id)
        prev_val = raw_regimes[mask].shift(1).dropna().mode()
        next_val = raw_regimes[mask].shift(-1).dropna().mode()
        replacement = prev_val[0] if not prev_val.empty else (next_val[0] if not next_val.empty else 0)
        filtered_regimes[mask] = replacement

# Find final regime change points
regimes = filtered_regimes
regime_changes = regimes[regimes != regimes.shift(1)].index.tolist()
if monthly.index[0] not in regime_changes:
    regime_changes.insert(0, monthly.index[0])

calendar = []
for start in regime_changes:
    data_up_to = monthly.loc[:start]
    data_clean = data_up_to.dropna()
    if data_clean.shape[0] < 2:
        continue  # skip if insufficient data

    mu_up_raw = data_clean.mean().values * 12
    mu_up = mu_up_raw - expense_vector

    try:
        lw_up = LedoitWolf().fit(data_clean.values)
        cov_up = lw_up.covariance_ * 12
    except ValueError:
        continue  # skip this regime start if LedoitWolf fails

    ef_up = efficient_frontier(cov_up, mu_up)
    idx_up, w_up = select_by_sigma(ef_up, voo_mu)
    calendar.append((start, regimes.loc[start], w_up))

df_calendar = pd.DataFrame([{
    'date': date,
    'regime': reg,
    **{f'w_{sym}': w[i] for i, sym in enumerate(etf_symbols)}
} for date, reg, w in calendar]).set_index('date')

print("\nForward-Looking Rebalancing at Regime Starts:")
print(df_calendar.head(len(calendar)))

# Show top-3 ETFs by weight for each regime
for idx, row in df_calendar.iterrows():
    weights = row.filter(like='w_').values
    top_idx = np.argsort(weights)[-3:][::-1]
    print(f"\nTop 3 ETFs at regime start {idx.date()} (Regime {int(row['regime'])}):")
    for i in top_idx:
        sym = etf_symbols[i]
        print(f"  {sym} ({etf_name_map.get(sym, 'Unknown')}): {weights[i]:.3%}")

df_calendar.to_csv("regime_rebalancing_weights.csv")
