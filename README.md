# Pairs Trading Backtest

Strategy using two tickers betting on mean reversion. 

## Idea

The core bet is mean reversion. If two assets share a stable relationship, a temporary divergence means they will eventually snap into alignment. Mean Reversion also means we're assuming the two stocks are stationary throughout the time series data. 





## Steps

1. **Pick a pair of stocks** 
2. **Fit a hedge ratio** — regress one stock on the other with OLS to find \(\beta\):

   Y_t = beta * X_t + alpha + Et
  
   The leftover Et is the spread that should mean-revert. This equation also estimates the long run relationship. 
3. **Apply Engle-Granger Cointegration** - Test residuals for stationarity using Augmented Dickey-Fuller Test (ADF) aka checking if p value < 0.05.

4. **Standardize** — convert the spread to a rolling z-score to know when it's stretched:

5. **Signal** — enter when the z-score crosses ±1.5, exit when it returns to 0.
   



## Learning Process 

- **Apple + Samsung** — I first selected these two since they were from the same sector which is technology. However, I ended up getting 0.0010 for \(\beta\), and 108.8647 for \alpha. The 0.0010 means that when Samsung moves by one unit, Apple moves by only about 0.001 units, making their relationship negligible.
- The 108.8647 alpha means that when Samsung is zero, Apple is worth about 109. 
- Found almost no real relationship. I failed because of currency/scale mismatch; the hedge ratio was meaningless.
- **WMT + TGT** — failed because they're not actually cointegrated; negative hedge ratio meant no mean-reversion.
- beta = -1.2029
- alpha = 209.0134
- **JPM + BAC** — though it seemed promising at first with the sharpe ratio I had gotten back, after the Engle-Granger Cointegration I found the two not to be cointegrated, meaning I couldn't bet on mean reversion. 



## Results

BEFORE COINTEGRATION: JPM/BAC over 2021–2026: the spread mean-reverts reliably, with the z-score swinging between roughly -3.4 and +3.2. The Total Return was 96.76, meaning the strategy gained 96.76 points over the period. The Sharpe ratio was 1.2(before transaction costs), and the Max drawdown was -40.22. 
I assumed that each trade will cost about 10 bps. After transaction costs, I ended with a Sharpe ratio of 1.14. 

AFTER: Learned that because JPM/BAC are not cointegrated, meaning they don't share a stable long run relationship, I shouldn't bet on them. The reason that they might not have cointegration even though they have the same currency and are of the same industry, the years that I chose, 2021-2026, might've had too many unpredictable factors like COVID-19. 



## Run It

```bash
pip install yfinance pandas numpy statsmodels matplotlib
python backtest.py
