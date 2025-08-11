@echo off
cd /d %~dp0
py -3 start.py scripts\mean_variance_optimal_portfolio.py || python start.py scripts\mean_variance_optimal_portfolio.py
pause
