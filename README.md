# ETF Portfolio Optimization Toolkit

*A reproducible showcase for tech-savvy economists and data-driven PMs.*

---

## 1 · Project Rationale
Modern portfolio research demands more than black-box backtests.  
This toolkit demonstrates **applied econometrics, robust optimization, and interactive data apps** in a single repo.  
Recruiters and hiring managers can clone the repo, run one notebook, and reproduce every figure in minutes.

---

## 2 · Feature Highlights
| Pillar | What it does | Why it matters |
| ------ | ------------ | -------------- |
| **Dynamic ETF universe** | Filters out sector-specific or redundant funds to keep a diversified core set | Mirrors real-world screening practices while remaining entirely open-source |
| **Robust mean-variance engine** | 50 efficient-frontier points plus bootstrap & 10 k Monte-Carlo stress tests | Shows command of statistical resampling and forward-looking risk estimation |
| **Sparse (L1) portfolios** | Optional `lambda_l1` parameter yields LASSO-style weight shrinkage | Demonstrates practical turnover control and interpretability |
| **Interactive Dash front-end** | Sliders + “Snap to VOO” button instantly match benchmark risk *or* return | Turns static analysis into a live demo for interviews & blog posts |

---

## 3 · Repository Layout



---

## 4 · Quick Start (Anaconda)

```bash
conda env create -f environment.yml
conda activate etf-opt

# Deep dive notebook
jupyter lab Optimal_Portfolio_Analysis.ipynb

# OR instant Dash demo
python mean_variance_optimal_portfolio.py
# → http://127.0.0.1:8050
