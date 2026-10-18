import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

# Download Fama-French five-factor data
ff_factors = pdr.DataReader(
    "F-F_Research_Data_5_Factors_2x3",
    "famafrench",
    start="2016-01-01",
    end="2025-12-31"
)[0]

# Fama-French returns are in percentage points, convert to decimals
ff_factors = ff_factors / 100

# Download monthly stock returns (example: Apple)
stock = yf.download("AAPL", start="2016-01-01",
                    end="2025-12-31", interval="1mo")
stock["Return"] = stock["Adj Close"].pct_change()
stock = stock.dropna()

# Align dates - Fama-French uses end-of-month dates
stock.index = stock.index.to_period("M")
ff_factors.index = ff_factors.index.to_period("M")
# Merge datasets
merged = stock[["Return"]].join(ff_factors, how="inner")

# Calculate excess returns (stock return minus risk-free rate)
merged["Excess_Return"] = merged["Return"] - merged["RF"]

# Define independent variables (the five factors)
X = merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
X = sm.add_constant(X)

# Define dependent variable (excess stock return)
y = merged["Excess_Return"]

# Run OLS regression
model = sm.OLS(y, X).fit()

print(model.summary())
print(f"\nAlpha (annualised): {model.params['const'] * 12:.4f}")
print(f"Market Beta: {model.params['Mkt-RF']:.4f}")
print(f"SMB Loading: {model.params['SMB']:.4f}")
print(f"HML Loading: {model.params['HML']:.4f}")
print(f"RMW Loading: {model.params['RMW']:.4f}")
print(f"CMA Loading: {model.params['CMA']:.4f}")
print(f"R-squared: {model.rsquared:.4f}")
