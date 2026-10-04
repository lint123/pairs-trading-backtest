
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS
import matplotlib.pyplot as plt


tickers = ["AAPL", "MSFT", "NVDA", "GOOGL",
           "AMZN", "META", "CRM", "ORCL", "INTC", "ADBE"]
data = yf.download(tickers, start="2021-01-01", end="2026-10-01")["Close"]
data = data.dropna(axis=1, thresh=int(len(data) * 0.9))
data = data.ffill()

# ---- Step  ​3: Cointegration loop ----
results = []
names = data.columns

for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        p_value = coint(data[a], data[b])[1]
        model = OLS(data[a], np.column_stack(
            [np.ones(len(data)), data[b]])).fit()
        beta = model.params.iloc[1]
        results.append({"pair": f"{a}-{b}", "ticker1": a,
                       "ticker2": b, "p_value": p_value, "beta": beta})

df = pd.DataFrame(results)
df = df[df["p_value"] < 0.05].sort_values("p_value")

# ---- Half-life function ----


def half_life(spread):
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    spread_lag = spread_lag[1:]
    spread_diff = spread_diff[1:]
    beta = OLS(spread_diff, spread_lag).fit().params.iloc[0]
    return -np.log(2) / beta


def hurst(ts):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(ts.subtract(ts.shift(lag)))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return 2 * poly[0]


    # ---- Compute half-life & Hurst for each surviving pair ----
for idx, row in df.iterrows():
    a, b = row["ticker1"], row["ticker2"]
    spread = data[a] - row["beta"] * data[b]
    df.at[idx, "half_life"] = half_life(spread)
    df.at[idx, "hurst"] = hurst(spread)

    # ---- Final filters ----
df = df[(df["half_life"] > 5) & (df["half_life"] < 30) & (df["hurst"] < 0.5)]
df = df.sort_values("p_value")
top_pairs = df.head(10)

# ---- Plot the top pairs ----
for _, row in top_pairs.iterrows():
    a, b = row["ticker1"], row["ticker2"]
    spread = data[a] - row["beta"] * data[b]
    z = (spread - spread.mean()) / spread.std()
    plt.plot(z)
    plt.axhline(2, color="red", linestyle="--")
    plt.axhline(-2, color="red", linestyle="--")
    plt.title(
        f"{a}-{b} | half-life: {row['half_life']:.1f}d | H: {row['hurst']:.2f}")
    plt.savefig(f"{a}-{b}.png")
    plt.close()

print(df[["pair", "p_value", "half_life", "hurst"]]).to_string()
print(f"Total pairs passing filters: {len(df)}")
