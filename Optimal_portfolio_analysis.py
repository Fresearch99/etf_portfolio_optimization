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
import datetime
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.tsa.regime_switching.api as smr

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
@np.vectorize
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
lw = LedoitWolf().fit(monthly.values)
annual_cov_shrunk = lw.covariance_ * 12

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

def select_by_sigma(frontier, target_sigma):
    if not frontier['sigma']: return None, None
    diffs = np.abs(np.array(frontier['sigma']) - target_sigma)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]

# ==============================================================================
# SECTION 1: LOAD EXOGENOUS DATA (YIELD CURVE & VIX)
# ==============================================================================
print("\n--- Section 1: Loading Exogenous Data for Regime Modeling ---")

start_date = monthly.index.min()
end_date = monthly.index.max()

# 1a. Fetch Yield Curve Data
def get_yield_curve(start, end):
    symbols = {'3M': "DGS3MO", '10Y': "DGS10"}
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
# SECTION 2: MARKOV REGIME-SWITCHING MODEL ESTIMATION
# ==============================================================================
print("\n--- Section 2: Estimating Markov Regime-Switching Models ---")

models = {}
for k in range(2, 5):
    print(f"Fitting model with {k} regimes...")
    try:
        # We use 'exog_tvtp' to make transition probabilities a function of our lagged variables
        mod = smr.MarkovRegression(
            endog,
            k_regimes=k,
            trend='c', # Constant term (mean) in each regime
            exog_tvtp=sm.add_constant(exog_final_lagged) # TVTP needs a constant
        )
        res = mod.fit(search_reps=10)
        models[k] = res
        print(f"  > BIC: {res.bic:.2f}, Log-Likelihood: {res.llf:.2f}")
    except Exception as e:
        print(f"  > Failed to fit model with {k} regimes: {e}")

# Select the best model based on BIC (lower is better)
best_k = min({k: v.bic for k, v in models.items()})
best_res = models[best_k]

print(f"\nBest model selected: {best_k} regimes (BIC = {best_res.bic:.2f})")

# Extract smoothed probabilities and identify the most likely regime for each month
smoothed_probs = best_res.smoothed_marginal_probabilities
regime_series = smoothed_probs.idxmax(axis=1).rename('regime')

# ==============================================================================
# SECTION 3: REGIME-SPECIFIC EFFICIENT FRONTIERS
# ==============================================================================
print("\n--- Section 3: Analyzing Regime-Specific Efficient Frontiers ---")

regime_frontiers = {}
regime_optimal_weights = {}

# Sort regimes by mean return for consistent labeling (e.g., Regime 0 is lowest return)
regime_means = best_res.params.filter(like='const').sort_values()
regime_order = regime_means.index.str.extract(r'\[(\d+)\]')[0].astype(int)
regime_map = {old: new for new, old in enumerate(regime_order)}

# Apply the new sorted order
regime_series = regime_series.map(regime_map)
smoothed_probs = smoothed_probs.rename(columns=regime_map).sort_index(axis=1)


plt.figure(figsize=(12, 7))
colors = plt.cm.jet(np.linspace(0, 1, best_k))

for i in range(best_k):
    print(f"\nAnalyzing Regime {i}...")
    in_regime = (regime_series == i)
    if in_regime.sum() < len(etf_symbols):
        print(f"  > Skipping Regime {i}, not enough data points ({in_regime.sum()})")
        continue

    # Estimate mu and cov for this regime using Ledoit-Wolf for robustness
    regime_monthly = monthly_final[in_regime]
    mu_regime = regime_monthly.mean().values * 12 - expense_vector
    lw_regime = LedoitWolf().fit(regime_monthly.values)
    cov_regime = lw_regime.covariance_ * 12

    # Generate efficient frontier for the regime
    ef_regime = efficient_frontier(cov_regime, mu_regime)
    regime_frontiers[i] = ef_regime

    # Find optimal portfolio matching VOO's overall volatility
    _, w_opt = select_by_sigma(ef_regime, voo_sigma_ann)
    if w_opt is not None:
        regime_optimal_weights[i] = w_opt
        print(f"  > Optimal portfolio for Regime {i} (matching VOO vol):")
        top_idx = np.argsort(w_opt)[-3:][::-1]
        for idx in top_idx:
            if w_opt[idx] > 0.01: # Show only significant weights
                print(f"    - {etf_symbols[idx]}: {w_opt[idx]:.1%}")
    else:
        print("  > Could not find an optimal portfolio for this regime.")
        regime_optimal_weights[i] = np.zeros_like(etf_symbols) # Fallback

    # Plot frontier
    if ef_regime['mu']:
        plt.plot(ef_regime['sigma'], ef_regime['mu'], label=f'Regime {i} Frontier', color=colors[i])
        # Mark the selected optimal portfolio
        if w_opt is not None:
            opt_sigma = np.sqrt(w_opt @ cov_regime @ w_opt)
            opt_mu = w_opt @ mu_regime
            plt.scatter(opt_sigma, opt_mu, marker='*', s=150, color=colors[i], zorder=5)

plt.scatter([voo_sigma_ann], [voo_mu_ann], color='black', marker='X', s=150, label='VOO (Overall)', zorder=5)
plt.title('Efficient Frontiers for Each Market Regime')
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

# We use the model's one-step-ahead predicted probabilities to form the portfolio for the next period.
# This simulates a real-world scenario where we use today's info to invest for tomorrow.
predicted_probs = best_res.predict()
predicted_probs.columns = predicted_probs.columns.map(regime_map) # Apply sorted regime map
predicted_probs = predicted_probs.sort_index(axis=1)

dynamic_weights = []
for t in range(len(monthly_final)):
    # Get the predicted probabilities for each regime at time t
    probs_t = predicted_probs.iloc[t]

    # Create a blended portfolio by weighting each regime's optimal portfolio by its probability
    blended_w = np.zeros(len(etf_symbols))
    for i in range(best_k):
        if i in regime_optimal_weights:
            blended_w += probs_t[i] * regime_optimal_weights[i]
    
    # Ensure weights sum to 1 (due to floating point errors or fallbacks)
    blended_w /= blended_w.sum()
    dynamic_weights.append(blended_w)

# Calculate the realized returns of this dynamic strategy
dynamic_port_returns = np.sum(np.array(dynamic_weights) * monthly_final[etf_symbols].values, axis=1)
dynamic_returns_series = pd.Series(dynamic_port_returns, index=monthly_final.index)

# Create cumulative return series for plotting
cum_dynamic = (1 + dynamic_returns_series).cumprod()
cum_voo_aligned = (1 + monthly_final['VOO']).cumprod()

# ==============================================================================
# SECTION 5: PERFORMANCE PLOTTING AND ANALYSIS
# ==============================================================================
print("\n--- Section 5: Performance Analysis ---")

# 5a. Plot Cumulative Performance
plt.figure(figsize=(12, 7))
plt.plot(cum_dynamic, label='Dynamic Regime-Switching Strategy', color='crimson')
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
    total_return = (returns_series.iloc[-1] - 1) * 100
    ann_return = ((1 + returns_series.mean())**12 - 1) * 100
    ann_vol = returns_series.std() * np.sqrt(12) * 100
    sharpe = ann_return / ann_vol # Assuming risk-free rate is 0 for simplicity
    
    # Max Drawdown
    cum_ret = (1 + returns_series).cumprod()
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret/peak - 1) * 100
    max_drawdown = drawdown.min()
    
    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": ann_return,
        "Annualized Volatility (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown
    }

metrics_dynamic = calculate_metrics(dynamic_returns_series)
metrics_voo = calculate_metrics(monthly_final['VOO'])

performance_df = pd.DataFrame({
    'Dynamic Strategy': metrics_dynamic,
    'VOO Benchmark': metrics_voo
}).T

print("\nPerformance Metrics Comparison:")
print(performance_df)

# 5c. Plot smoothed probabilities with VOO returns
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
smoothed_probs.plot(ax=axes[0], kind='area', stacked=True, colormap='viridis')
axes[0].set_title('Smoothed Probabilities of Each Regime Over Time')
axes[0].set_ylabel('Probability')
axes[0].legend(title='Regime', loc='upper left')

monthly_final['VOO'].plot(ax=axes[1], color='black', label='VOO Monthly Return')
axes[1].set_title('VOO Monthly Returns')
axes[1].set_ylabel('Return')
axes[1].grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



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
import datetime
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.tsa.regime_switching.api as smr

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
@np.vectorize
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
lw = LedoitWolf().fit(monthly.values)
annual_cov_shrunk = lw.covariance_ * 12

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

def select_by_sigma(frontier, target_sigma):
    if not frontier['sigma']: return None, None
    diffs = np.abs(np.array(frontier['sigma']) - target_sigma)
    idx = diffs.argmin()
    return idx, frontier['weights'][idx]

# ==============================================================================
# SECTION 1: LOAD EXOGENOUS DATA (YIELD CURVE & VIX)
# ==============================================================================
print("\n--- Section 1: Loading Exogenous Data for Regime Modeling ---")

start_date = monthly.index.min()
end_date = monthly.index.max()

# 1a. Fetch Yield Curve Data
def get_yield_curve(start, end):
    symbols = {'3M': "DGS3MO", '10Y': "DGS10"}
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

models = {}
for k in range(2, 5):
    print(f"Fitting model with {k} regimes...")
    try:
        # **NEW**: Added `switching_variance=True`
        mod = smr.MarkovRegression(
            endog,
            k_regimes=k,
            trend='c',
            switching_variance=True, # Allows volatility to be regime-specific
            exog_tvtp=sm.add_constant(exog_final_lagged)
        )
        res = mod.fit(search_reps=20) # More complex model, more search reps
        models[k] = res
        print(f"  > BIC: {res.bic:.2f}, Log-Likelihood: {res.llf:.2f}")
    except Exception as e:
        print(f"  > Failed to fit model with {k} regimes: {e}")

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
# SECTION 3: REGIME-SPECIFIC EFFICIENT FRONTIERS (WITH MEAN SHRINKAGE)
# ==============================================================================
print("\n--- Section 3: Analyzing Regime-Specific Efficient Frontiers ---")
print("NOTE: Using Bayesian shrinkage for mean returns within each regime.")

# **NEW**: Helper function for Bayesian mean shrinkage
def shrink_means_bayesian(mu, cov, tau=0.05):
    """
    Shrinks a vector of sample means (mu) towards the grand mean using a
    Bayesian/Black-Litterman-style formula.
    - tau: A scalar representing the uncertainty in the prior. A smaller tau
           leads to stronger shrinkage towards the grand mean.
    """
    prior_mu = np.mean(mu) * np.ones_like(mu)
    
    # Using shrinkage for covariance of the prior is more stable
    lw = LedoitWolf().fit(np.diag(np.diag(cov))) # Prior assumes no correlation
    cov_prior = lw.covariance_ / tau
    
    # Posterior calculation (numerically stable version)
    # inv(inv(A) + inv(B)) @ (inv(A)@a + inv(B)@b)
    # Let C1 = inv(cov_prior), C2 = inv(cov)
    # mu_post = inv(C1 + C2) @ (C1@prior_mu + C2@mu)
    try:
        C1 = np.linalg.inv(cov_prior)
        C2 = np.linalg.inv(cov)
        cov_post = np.linalg.inv(C1 + C2)
        mu_post = cov_post @ (C1 @ prior_mu + C2 @ mu)
        return mu_post
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular
        print("  > Warning: Singular matrix in mean shrinkage. Returning sample mean.")
        return mu

regime_frontiers = {}
regime_optimal_weights = {}

plt.figure(figsize=(12, 7))
colors = plt.cm.plasma(np.linspace(0, 1, best_k))

for i in range(best_k):
    print(f"\nAnalyzing Regime {i}...")
    in_regime = (regime_series == i)
    if in_regime.sum() < len(etf_symbols):
        print(f"  > Skipping Regime {i}, not enough data points ({in_regime.sum()})")
        continue

    regime_monthly = monthly_final[in_regime]
    
    # 1. Estimate sample moments
    mu_regime_sample = regime_monthly.mean().values * 12 - expense_vector
    lw_regime = LedoitWolf().fit(regime_monthly.values)
    cov_regime = lw_regime.covariance_ * 12
    
    # 2. **NEW**: Apply mean shrinkage
    mu_regime_shrunk = shrink_means_bayesian(mu_regime_sample, cov_regime)

    # 3. Generate efficient frontier with shrunk means
    ef_regime = efficient_frontier(cov_regime, mu_regime_shrunk)
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
            opt_mu = w_opt @ mu_regime_shrunk # Use shrunk mu for plotting
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

predicted_probs = best_res.predict()
predicted_probs.columns = predicted_probs.columns.map(regime_map) # Apply sorted regime map
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
    total_return = (returns_series.iloc[-1] - 1) * 100
    ann_return = ((1 + returns_series.mean())**12 - 1) * 100
    ann_vol = returns_series.std() * np.sqrt(12) * 100
    sharpe = ann_return / ann_vol
    cum_ret = (1 + returns_series).cumprod()
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret/peak - 1) * 100
    max_drawdown = drawdown.min()
    return {"Total Return (%)": total_return, "Annualized Return (%)": ann_return,
            "Annualized Volatility (%)": ann_vol, "Sharpe Ratio": sharpe, "Max Drawdown (%)": max_drawdown}

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
# SECTION 2: MARKOV REGIME-SWITCHING MODEL ESTIMATION (WITH SWITCHING VARIANCE)
# ==============================================================================
print("\n--- Section 2: Estimating Markov Regime-Switching Models ---")
print("NOTE: Now allowing both Mean and Variance to switch between regimes.")

models = {}
for k in range(2, 5):
    print(f"Fitting model with {k} regimes...")
    try:
        mod = smr.MarkovRegression(
            endog,
            k_regimes=k,
            trend='c',
            switching_variance=True, # Allows volatility to be regime-specific
            exog_tvtp=sm.add_constant(exog_final_lagged)
        )
        res = mod.fit(search_reps=20) 
        models[k] = res
        print(f"  > BIC: {res.bic:.2f}, Log-Likelihood: {res.llf:.2f}")
    except Exception as e:
        print(f"  > Failed to fit model with {k} regimes: {e}")

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
# SECTION 3: REGIME-SPECIFIC EFFICIENT FRONTIERS (WITH MEAN SHRINKAGE)
# ==============================================================================
print("\n--- Section 3: Analyzing Regime-Specific Efficient Frontiers ---")
print("NOTE: Using Bayesian shrinkage for mean returns within each regime.")

def shrink_means_bayesian(mu, cov, tau=0.05):
    """
    Shrinks a vector of sample means (mu) towards the grand mean using a
    Bayesian/Black-Litterman-style formula.
    - tau: A scalar representing the uncertainty in the prior. A smaller tau
           leads to stronger shrinkage towards the grand mean.
    """
    prior_mu = np.mean(mu) * np.ones_like(mu)
    
    # Using shrinkage for covariance of the prior is more stable
    lw = LedoitWolf().fit(np.diag(np.diag(cov))) # Prior assumes no correlation
    cov_prior = lw.covariance_ / tau
    
    # Posterior calculation (numerically stable version)
    # inv(inv(A) + inv(B)) @ (inv(A)@a + inv(B)@b)
    # Let C1 = inv(cov_prior), C2 = inv(cov)
    # mu_post = inv(C1 + C2) @ (C1@prior_mu + C2@mu)
    try:
        C1 = np.linalg.inv(cov_prior)
        C2 = np.linalg.inv(cov)
        cov_post = np.linalg.inv(C1 + C2)
        mu_post = cov_post @ (C1 @ prior_mu + C2 @ mu)
        return mu_post
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular
        print("  > Warning: Singular matrix in mean shrinkage. Returning sample mean.")
        return mu

regime_frontiers = {}
regime_optimal_weights = {}

plt.figure(figsize=(12, 7))
colors = plt.cm.plasma(np.linspace(0, 1, best_k))

for i in range(best_k):
    print(f"\nAnalyzing Regime {i}...")
    in_regime = (regime_series == i)
    if in_regime.sum() < len(etf_symbols):
        print(f"  > Skipping Regime {i}, not enough data points ({in_regime.sum()})")
        continue

    regime_monthly = monthly_final[in_regime]
    
    # 1. Estimate sample moments
    mu_regime_sample = regime_monthly.mean().values * 12 - expense_vector
    lw_regime = LedoitWolf().fit(regime_monthly.values)
    cov_regime = lw_regime.covariance_ * 12
    
    # 2. **NEW**: Apply mean shrinkage
    mu_regime_shrunk = shrink_means_bayesian(mu_regime_sample, cov_regime)

    # 3. Generate efficient frontier with shrunk means
    ef_regime = efficient_frontier(cov_regime, mu_regime_shrunk)
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
            opt_mu = w_opt @ mu_regime_shrunk # Use shrunk mu for plotting
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

predicted_probs = best_res.predict()
predicted_probs.columns = predicted_probs.columns.map(regime_map) # Apply sorted regime map
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
    total_return = (returns_series.iloc[-1] - 1) * 100
    ann_return = ((1 + returns_series.mean())**12 - 1) * 100
    ann_vol = returns_series.std() * np.sqrt(12) * 100
    sharpe = ann_return / ann_vol
    cum_ret = (1 + returns_series).cumprod()
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret/peak - 1) * 100
    max_drawdown = drawdown.min()
    return {"Total Return (%)": total_return, "Annualized Return (%)": ann_return,
            "Annualized Volatility (%)": ann_vol, "Sharpe Ratio": sharpe, "Max Drawdown (%)": max_drawdown}

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

