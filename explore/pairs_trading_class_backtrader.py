# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import quantstats
import collections
import matplotlib.pyplot as plt
import pandas as pd
from src.load_data import fetch_crypto_data, fetch_data
from src.pairs_trading_functions import*
from binance import Client
from src.api_key_secret import api_key, api_secret, path_zert

#%%

from backtrader.indicators.hurst import Hurst


class PairsTrading(bt.Strategy):
    params = (
        ("window", None),
        ("std_dev", None),
        ("size", None)
    )

    def __init__(self):
        self.data_a = self.datas[0].close
        self.data_b = self.datas[1].close
        self.hedge_ratio = None
        self.window = self.params.window
        self.zscore = None
        self.spread_history = collections.deque(maxlen=self.params.window)
        self.upper_bound = self.params.std_dev
        self.lower_bound = -self.params.std_dev
        self.size = self.params.size
        self.trade_size = 0
        self.equity = None

        self.spread_history_full = []
        self.zscore_history = []
        self.hedge_ratio_history = []

        self.ols_slope = btind.OLS_Slope_InterceptN(self.data_a, self.data_b, period=self.params.window)
        self.hurst_exponent = None
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print("{} {}".format(dt.isoformat(), txt))


    def calc_hedge_ratio(self):

        hedge_ratio = self.ols_slope.slope[0]
        spread = self.data_a[0] - (hedge_ratio * self.data_b[0])
        #print(f'spread: ' +str(spread))
        self.spread_history.append(spread)
        spread_mean = pd.Series(self.spread_history).rolling(self.params.window).mean().iloc[-1]
        spread_std_dev = pd.Series(self.spread_history).rolling(self.params.window).std().iloc[-1]
        self.zscore = (spread - spread_mean) / spread_std_dev
        self.hedge_ratio = hedge_ratio

        self.spread_history_full.append(spread)
        self.zscore_history.append(self.zscore)
        self.hedge_ratio_history.append(hedge_ratio)

        # calc hurst exponent
        if len(self.spread_history) >= self.params.window:
            lags = range(2, self.params.window // 2)

            # Calculate the array of the variances of the lagged differences
            tau = [np.sqrt(np.abs(pd.Series(self.spread_history) - pd.Series(self.spread_history).shift(lag)).dropna().var()) for lag in lags]

            # Use a linear fit to estimate the Hurst Exponent
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            self.hurst_exponent = poly[0] * 2




    def start(self):
        self.equity = self.broker.get_cash()


    def next(self):

        self.equity = self.broker.get_value()
        self.trade_size = self.equity * self.params.size / self.data_a[0]
        self.calc_hedge_ratio()
        print("Hurst: " + str(self.hurst_exponent))

        #if len(self.spread_history) > self.params.window:
            #hurst = hurst_2(pd.Series(self.spread_history)[-self.params.window:])
            #print(hurst)
        #self.hurst(self.spread_history[-self.params.window:])
        #self.log("Hurst: {}".format(self.hurst[0][-1]))
        #self.hurst = hurst_2(pd.Series(self.spread_history_full[-self.params.window:]))
        #print(self.hurst)

        # Check if there is already an open trade
        if self.getposition().size == 0:
            if self.zscore < self.lower_bound:
                # Buy the spread
                self.log("BUY SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                self.log("Z-SCORE: {}".format(self.zscore))
                self.log("Portfolio Value: {}".format(self.equity))
                self.order_target_size(self.datas[0], self.trade_size)
                self.order_target_size(self.datas[1], -self.hedge_ratio * self.trade_size)

            elif self.zscore > self.upper_bound:
                # Sell the spread
                self.log("SELL SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                self.log("Z-SCORE: {}".format(self.zscore))
                self.log("Portfolio Value: {}".format(self.equity))
                self.order_target_size(self.datas[0], -self.trade_size)
                self.order_target_size(self.datas[1], self.hedge_ratio * self.trade_size)


        # If there is an open trade, wait until the zscore crosses zero
        elif self.getposition().size > 0 and self.zscore > 0:
            self.log("CLOSE LONG SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
            self.log("Z-SCORE: {}".format(self.zscore))
            self.log("Portfolio Value: {}".format(self.equity))
            self.order_target_size(self.datas[0], 0)
            self.order_target_size(self.datas[1], 0)


        elif self.getposition().size < 0 and self.zscore < 0:
            self.log("CLOSE SHORT SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
            self.log("Z-SCORE: {}".format(self.zscore))
            self.log("Portfolio Value: {}".format(self.equity))
            self.order_target_size(self.datas[0], 0)
            self.order_target_size(self.datas[1], 0)



#%%

# check in next if cointegration is still valide
# if not are decying -> look for new coointegrated pairs

if __name__ == "__main__":
    days = 180
    cerebro = bt.Cerebro()

    # Fetch data and find cointegrated pairs
    client = Client(api_key, api_secret)
    data = fetch_crypto_data(20, days, client)
    pairs = find_cointegrated_pairs_2(data)

    window = int(pairs.iloc[0,-1])
    #window = 1000
    std_dev = 1
    size = 0.01

    # Choose the pair with the smallest p-value
    tickers_pairs = pairs.iloc[0, 0:2]
    print(f'trading pair: ' + str(tickers_pairs))

    # Fetch data for the chosen pair
    data_df0 = fetch_data(tickers_pairs[0], '1h', str(days * 24), client)
    data_df1 = fetch_data(tickers_pairs[1], '1h', str(days * 24), client)
    data0 = bt.feeds.PandasData(dataname=pd.DataFrame(data_df0))
    data1 = bt.feeds.PandasData(dataname=pd.DataFrame(data_df1))
    cerebro.adddata(data0)
    cerebro.adddata(data1)

    # Add the strategy
    cerebro.addstrategy(PairsTrading, window=window, std_dev=std_dev, size=size)

    # Set the commission and the starting cash
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.setcash(100000)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

    # Run the backtest
    results = cerebro.run()

    # Print the final portfolio value
    final_value = cerebro.broker.getvalue()
    print("Final portfolio value: ${}".format(final_value))

    # Get the analyzers and print the results
    trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
    print("Starting cash: ${}".format(cerebro.broker.startingcash))
    print("Ending cash: ${}".format(cerebro.broker.getvalue()))
    print("Total return: {:.2f}%".format(100*((cerebro.broker.getvalue()/cerebro.broker.startingcash)-1)))
    print("Half-Live: " + str(int(pairs.iloc[0,-1])) + " hours")
    print("Number of trades: {}".format(trade_analyzer.total.closed))
    print("Winning Trades:", results[0].analyzers.trade_analyzer.get_analysis()['won']['total'])
    print("Losing Trades:", results[0].analyzers.trade_analyzer.get_analysis()['lost']['total'])

    # Get the strategy instance
    strategy_instance = results[0]

    # Plot the spread, zscore, and hedge ratio
    plt.subplot(3, 1, 1)
    plt.plot(strategy_instance.spread_history_full)
    plt.title(f'Spread {list(tickers_pairs)}')

    plt.subplot(3, 1, 2)
    plt.plot(strategy_instance.zscore_history)
    plt.axhline(strategy_instance.upper_bound, color='r')
    plt.axhline(strategy_instance.lower_bound, color='r')
    plt.title("Z-score")
    plt.legend(["Z-score"])

    plt.subplot(3, 1, 3)
    plt.plot(strategy_instance.hedge_ratio_history)
    plt.title("Hedge ratio")
    plt.legend(["Hedge ratio"])

    plt.tight_layout()
    plt.show()

    # create cerebro chart
    cerebro.plot(iplot=True, volume=False)

    # create quantstats charts & statistics html
    portfolio_stats = strategy_instance.analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)

    quantstats.reports.html(returns, output='stats.html', title='Backtrade Pairs')


#%%

def find_cointegrated_pairs_2(data):
    ''' find from a list cointegrated pairs'''
    n = data.shape[1]
    keys = data.keys()
    pvalue_matrix = np.ones((n, n))
    pairs = []

    # Loop through each combination of assets
    for i in range(n):
        for j in range(i+1, n):
            S1 = pd.to_numeric(data[keys[i]])
            S2 = pd.to_numeric(data[keys[j]])

            # Test for cointegration
            result = ts.coint(S1, S2)
            pvalue = result[1]

            # Store p-value in matrix
            pvalue_matrix[i, j] = pvalue

            # Add cointegrated pair to list (if p-value is less than 0.05)
            if pvalue < 0.05:

                model = sm.OLS(S1, S2)
                results = model.fit()
                hedgeRatio = results.params
                z = S1 - hedgeRatio[0] * S2
                prevz = z.shift()
                dz = z-prevz
                dz = dz[1:,]
                prevz = prevz[1:,]
                model2 = sm.OLS(dz, prevz-np.mean(prevz))
                results2 = model2.fit()
                theta = results2.params
                half_life = -np.log(2)/theta


                pairs.append((keys[i], keys[j], pvalue, half_life.values))

    # Sort cointegrated pairs by p-value in ascending order
    pairs.sort(key=lambda x: x[2])

    return pd.DataFrame(pairs, columns=['Asset 1', 'Asset 2', 'P-value', 'Half Life'])


def hurst_2(df_series):
    """Returns the Hurst exponent of the time series vector ts"""
    # df_series = df_series if not isinstance(df_series, pd.Series) else df_series.to_list()
    # Create the range of lag values
    lags = range(2, 10)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt((df_series - df_series.shift(-lag)).std()) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2

#%%

window = 10

test2 = hurst_2(pd.Series(strategy_instance.spread_history_full)[-window:])



#%%
