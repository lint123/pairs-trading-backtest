import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
lookback_window = 20
data = yf.download(['JPM', 'BAC'], start='2021-01-01',
                   end='2026-01-01')
close = data['Close'].dropna()
X = sm.add_constant(close['BAC'])
model = sm.OLS(close['JPM'], X).fit()
beta = model.params['BAC']
alpha = model.params['const']
spread = close['JPM'] - (beta * close['BAC'] + alpha)
print(f"beta: {beta:.4f}")
print(f"alpha: {alpha:.4f}")
plt.figure(figsize=(5, 3))
plt.plot(spread, label='Spread')
plt.title('JP Morgan and Bank of America')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.grid(True)
plt.show()
close['Spread'] = close['JPM'] - (beta * close['BAC'] + alpha)
close['Spread_Mean'] = close['Spread'].rolling(window=lookback_window).mean()
close['Spread_std'] = close['Spread'].rolling(window=lookback_window).std()
Z_Score = (close['Spread'] - close['Spread_Mean'])/close['Spread_std']
print(Z_Score.describe())
plt.plot(Z_Score, label='Z-Score')
plt.ylabel('Standard Deviations')
plt.xlabel('Year')
plt.title('Rolling Z Score & Statistical Thresholds')
plt.grid(True)
plt.show()
