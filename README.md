# Pairs Trading Backtest

A strategy built from scratch in Python: find two stocks/tickers that move together, bet that when they drift apart, they snap back.

## Idea

The core bet is mean reversion. If two assets share a stable relationship, a temporary divergence means they will eventually snap into alignment. 

.



## Steps

1. **Pick a pair** — same sector, same market, same currency.
2. **Fit a hedge ratio** — regress one stock on the other with OLS to find \(\beta\):
   \[
   Y_t = \beta X_t + \alpha + \varepsilon_t
   \]
   The leftover \(\varepsilon_t\) is the spread that should mean-revert.
.
. **Standardize** — convert the spread to a rolling z-score to know when it's stretched:
   \[
   z_t = \frac{S_t - \mu}{\sigma}
   \]
4. **Signal** — enter when the z-score crosses ±1.5, exit when it returns to 0.



## Learning Process 

- **Apple + Samsung** — I first selected these two since they were from the same sector which is technology. However, I ended up getting 0.0010 for \(\beta\), and 108.8647 for \alpha. The 0.0010 means that when Samsung moves by one unit, Apple moves by only about 0.001 units, making their relationship negligible.
- The 108.8647 alpha means that when Samsung is zero, Apple is worth about 109. 
- Found almost no real relationship. I failed because of currency/scale mismatch; the hedge ratio was meaningless.
- **WMT + TGT** — failed because they're not actually cointegrated; negative hedge ratio meant no mean-reversion.
- \(\beta\) = -1.2029
- \alpha = 209.0134
.
- **JPM + BAC** — worked: the z-score oscillates between -3 and +3 around zero, a mean-reverting signal.



## Results

JPM/BAC over 2021–2026: the spread mean-reverts reliably, with the z-score swinging between roughly -3.4 and +3.2. The Total Return was 96.76, meaning the strategy gained 96.76 points over the period. The Sharpe ratio was 1.2(before transaction costs), and the Max drawdown was -40.22. 
I assumed that each trade will cost about 10 bps. After transaction costs, I ended with a Sharpe ratio of 1.14. 



## Run It

```bash
pip install yfinance pandas numpy statsmodels matplotlib
python backtest.py
