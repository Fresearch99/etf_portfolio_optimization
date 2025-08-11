# ETF Portfolio Optimization Toolkit

How can I optimize my investments?  That is one of the core questions in finance—and the motivation for this repo.  We walk through multiple portfolio-optimization approaches to find which combination of Vanguard ETFs is optimal for an individual investor aiming to outperform **VOO** (the S\&P 500 index ETF).  The main story is presented through a [**narrative Jupyter notebook**](notebooks/Optimal_Portfolio_Analysis.ipynb), and we ultimately find that a simple **Markowitz mean-variance optimization** performs best.  In our tests (as of Aug 2025) we can, with periodic refits on recent data, outperform VOO by **>3% on average out-of-sample** and improve the **Sharpe ratio by \~25%** over the evaluation period.  A small **one-click Dash demo** lets you explore the efficient frontier and select portfolios by target return or risk.

---

## How to Clone This Repository

Choose one method and replace the URL with your repo path if different.

### HTTPS

```bash
git clone https://github.com/Fresearch99/portfolio_optimization.git
cd portfolio_optimization
```

### SSH (requires configured SSH keys)

```bash
git clone git@github.com:Fresearch99/portfolio_optimization.git
cd portfolio_optimization
```

> macOS tip: if a `.command` file doesn’t run on double-click, make it executable:
>
> ```bash
> chmod +x "Run Mean-Variance.command" "Stop App.command"
> ```

---

## What the Notebook Does (Step by Step)

1. **Load & Clean**

   * Scrape ETF metadata from Vanguard’s website and load monthly return data from `yfinance`.
   * Save ETF metadata and monthly returns to `data/` (and load from there if scraping/fetching is unavailable).
   * Drop ETFs with short histories (e.g., < 10 years) and rows with remaining NAs.
   * Keep a broad-market subset (optionally remove sector-specific/overlapping ETFs).

2. **Estimate Inputs**

   * Compute **annualized mean returns** (monthly mean × 12) **net of expense ratios**.
   * Compute **annualized covariance** (monthly covariance × 12).
   * Explore shrinkage variants for means/covariances (e.g., **James-Stein**, **Ledoit–Wolf**).

3. **Optimize (Mean–Variance)**

   * Build the efficient frontier by sweeping a grid of target returns.
   * Solve a **long-only, fully invested** quadratic program; optionally add **L1 regularization**.
   * Find optimal portfolios that match VOO’s mean or variance.
   * Visualize the frontier and display the **top-3 ETF weights**.

4. **Advanced Methods & Robustness**

   * Implement alternatives to test robustness:

     * Resampled Efficient Frontier
     * Rolling Estimation
     * Black–Litterman
     * Risk Parity
     * Hierarchical Risk Parity
     * DCC-GARCH
     * Markov regime switching (with VIX as an exogenous predictor)
   * Compare performance in-sample and via Monte Carlo simulation.

5. **Out-of-Sample Testing**

   * For the two best performers (simple mean-variance optimization and the Markov regime-switching model), split data into train/test.
   * Refit periodically as new data arrive.
   * Compare and visualize both models against VOO over the out-of-sample period.

6. **Reproduce the Optimal Portfolio (Conclusion)**

   * Provide a **one-click Dash app** (`start.py` + `scripts/mean_variance_optimal_portfolio.py`) that loads the same data and renders the frontier interactively based on the best-performing approach-simple mean-variance optimization.

---

## Results (summary)

* **The best-performing portfolio is produced by simple mean-variance optimization.**
* The optimal portfolio is often a combination of a higher-risk growth ETF and a Treasury ETF—**diversification works**.
* **VOO** is a strong benchmark; most advanced methods do **not** consistently beat it.  In our data, the mean-variance optimal portfolio that matches VOO’s risk delivered **\~19.7%** average return over the last three years vs **\~16.6%** for VOO (based on the period and dataset as used in the notebook).
* Regime-switching models are sensitive to the sample and macro regime. Our Markov regime-switching model **outperforms in-sample** but **fails** during the recent high-interest-rate out-of-sample period.

**Practical conclusion:** the core mean-variance analysis is packaged in the **one-click Dash app**, which visualizes the frontier and top weights for different risk/return preferences.

---

## Data & Assumptions

`data/` contains cleaned ETF metadata, monthly and weekly ETF returns, and FRED data from a run of the main notebook (Aug 2025).  For the Dash app, two files matter:

* `df_etf_metadata.csv` — columns **`Symbol`**, **`Fund name`**, **`Expense ratio`**
* `returns_monthly.csv` — wide format with a date index; columns are ETF tickers; values are **monthly returns** in decimals (e.g., `0.0123` for 1.23%)

The notebook comments out the save commands.  If you want to refresh with more recent data, **uncomment** the lines that save the CSV files.

Similarly, `outputs/` contains figures produced by the notebook. Save commands are commented out and can be **uncommented** to regenerate with new data.

---

## What’s in the Repo

```
.
├─ data/
│  ├─ df_etf_metadata.csv
│  ├─ fred_data.csv                         # used by the Markov regime-switching model
│  ├─ returns_monthly.csv
│  └─ returns_weekly.csv                    # used by the DCC-GARCH model
├─ docs/
│  └─ Optimal_Portfolio_Analysis.html       # read-only HTML export of the main notebook
├─ notebooks/
│  └─ Optimal_Portfolio_Analysis.ipynb      # main narrative notebook
├─ outputs/                                 # figures produced by the notebook
├─ scripts/
│  └─ mean_variance_optimal_portfolio.py    # Dash UI (frontier + top weights)
├─ environment.yml                          # Conda env for notebooks (JupyterLab/VS Code)
├─ requirements.txt                         # lean runtime deps for the Dash app
├─ start.py                                 # venv + install + run for the Dash app
├─ Run Mean-Variance.bat                    # one-click launch (Windows)
├─ Run Mean-Variance.command                # one-click launch (macOS)
├─ Stop App.bat                             # one-click stop (Windows)
└─ Stop App.command                         # one-click stop (macOS)
```

---

## How to Run

### Notebook

```bash
mamba env create -f environment.yml      # or: conda env create -f environment.yml
conda activate etf-optimizer
python -m ipykernel install --user --name etf-optimizer --display-name "ETF Optimizer"
```

Open **notebooks/Optimal\_Portfolio\_Analysis.ipynb**, select the **ETF Optimizer** kernel, then run cells top-to-bottom.

### One-Click Dash

```bash
python start.py scripts/mean_variance_optimal_portfolio.py
```

* Creates `./.venv` (if needed), installs `requirements.txt`, and opens **[http://127.0.0.1:8050/](http://127.0.0.1:8050/)**.
* Stop with **Ctrl+C** in the terminal. The app also tries to stop on tab close (watchdog).

Or double-click `Run Mean-Variance.bat` (Windows) / `Run Mean-Variance.command` (macOS).  To stop, double-click `Stop App.bat` (Windows) or `Stop App.command` (macOS).

---

## License

MIT license — see [`LICENSE`](LICENSE).

