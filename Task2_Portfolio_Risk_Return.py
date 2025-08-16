
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Select stocks and download data
stocks = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
data = yf.download(stocks, start="2020-01-01", end="2025-01-01")["Close"]

# Step 2: Calculate daily returns
daily_returns = data.pct_change().dropna()

# Step 3: Expected return (mean of daily returns)
expected_returns = daily_returns.mean()
print("Expected Returns (Daily):")
print(expected_returns)

# Annualize expected return
annual_expected_returns = expected_returns * 252
print("\nExpected Returns (Annualized):")
print(annual_expected_returns)

# Step 4: Portfolio metrics (Equal weights)
weights = np.array([1/3, 1/3, 1/3])

# Portfolio expected return
portfolio_return = np.dot(annual_expected_returns, weights)
print("\nPortfolio Expected Return (Annualized):", portfolio_return)

# Portfolio risk (standard deviation)
cov_matrix = daily_returns.cov() * 252  # Annualized covariance matrix
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
portfolio_std = np.sqrt(portfolio_variance)
print("Portfolio Risk (Std Dev):", portfolio_std)

# Step 5: Sharpe Ratio (assuming 4% risk-free rate)
risk_free_rate = 0.04
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
print("Portfolio Sharpe Ratio:", sharpe_ratio)

# Step 6: Visualization
plt.figure(figsize=(10,6))
plt.scatter(daily_returns.std()*np.sqrt(252), annual_expected_returns, alpha=0.7)
for i, txt in enumerate(stocks):
    plt.annotate(txt, (daily_returns.std().values[i]*np.sqrt(252), annual_expected_returns.values[i]))
plt.scatter(portfolio_std, portfolio_return, color="red", label="Portfolio")
plt.xlabel("Risk (Std Dev)")
plt.ylabel("Expected Return")
plt.title("Portfolio Risk-Return Analysis")
plt.legend()
plt.show()
