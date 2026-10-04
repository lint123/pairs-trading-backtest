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

position = 0

daily_returns = []
cost_bps = 0.001  # 10 bps per side
# enter long - betting spread will go up
# enter short - betting spread will go down
for i in range(1, len(Z_Score)):
    z = Z_Score.iloc[i]
    prev_z = Z_Score.iloc[i-1]
    spread_today = close['Spread'].iloc[i]
    spread_yesterday = close['Spread'].iloc[i-1]
    # enter signal
    if position == 1:
        daily_returns.append(spread_today - spread_yesterday)
    elif position == -1:
        daily_returns.append(spread_yesterday - spread_today)

    if position == 1 and z <= -0.5:
        position = 0
        if daily_returns:
            daily_returns[-1] -= cost_bps * abs(spread_today)
    elif position == -1 and z >= 0.5:
        position = 0
        if daily_returns:
            daily_returns[-1] -= cost_bps * abs(spread_today)

    elif position == 0 and prev_z < -1.5 and z >= -1.5:
        position = 1  # long: spread stretched low, bet it rises
        if daily_returns:
            daily_returns[-1] -= cost_bps * abs(spread_today)
    elif position == 0 and prev_z > 1.5 and z <= 1.5:
        position = -1  # short: spread stretched high, bet it falls
        if daily_returns:
            daily_returns[-1] -= cost_bps * abs(spread_today)

returns = pd.Series(daily_returns)

total_return = returns.sum()
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
cumulative = returns.cumsum()
peak = cumulative.cummax()
drawdown = cumulative - peak
max_drawdown = drawdown.min()


print(f"Total return: {total_return:.4f}")
print(f"Sharpe ratio: {sharpe:.4f}")
print(f"Max drawdown: {max_drawdown:.4f}")
