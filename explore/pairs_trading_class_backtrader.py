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
from src.pairs_trading_functions import find_cointegrated_pairs
from src.load_data import fetch_crypto_data, fetch_data
from binance import Client
from src.api_key_secret import api_key, api_secret, path_zert


#%%

class PairsTrading(bt.Strategy):
   """ params = (
        ("window", 120),
        ("std_dev", 1),
        ("size", 10000)
    )
"""
    def __init__(self, window, std_dev, size):
        self.data_a = self.datas[0].close
        self.data_b = self.datas[1].close
        self.hedge_ratio = None
        self.window = window
        self.zscore = None
        self.spread_history = collections.deque(maxlen=self.window)
        self.upper_bound = std_dev
        self.lower_bound = -std_dev
        self.size = size

        self.spread_history_full = []
        self.zscore_history = []
        self.hedge_ratio_history = []

        self.ols_slope = btind.OLS_Slope_InterceptN(self.data_a, self.data_b, period=self.window)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print("{} {}".format(dt.isoformat(), txt))

    def calc_hedge_ratio(self):

        hedge_ratio = self.ols_slope.slope[0]
        spread = self.data_a[0] - (hedge_ratio * self.data_b[0])
        #print(f'spread: ' +str(spread))
        self.spread_history.append(spread)
        spread_mean = pd.Series(self.spread_history).rolling(self.window).mean().iloc[-1]
        spread_std_dev = pd.Series(self.spread_history).rolling(self.window).std().iloc[-1]
        self.zscore = (spread - spread_mean) / spread_std_dev
        self.hedge_ratio = hedge_ratio

        self.spread_history_full.append(spread)
        self.zscore_history.append(self.zscore)
        self.hedge_ratio_history.append(hedge_ratio)

        #print(f'zscore: ' +str((spread - spread_mean) / spread_std_dev))


    def next(self):
        self.calc_hedge_ratio()

        # Check if there is already an open trade
        if self.position.size == 0:
            if self.zscore < self.lower_bound:
                # Buy the spread
                self.log("BUY SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                self.log("Z-SCORE: {}".format(self.zscore))
                self.order_target_size(self.datas[0], self.size)
                self.order_target_size(self.datas[1], -self.hedge_ratio * self.size)
            elif self.zscore > self.upper_bound:
                # Sell the spread
                self.log("SELL SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                self.log("Z-SCORE: {}".format(self.zscore))
                self.order_target_size(self.datas[0], -self.size)
                self.order_target_size(self.datas[1], self.hedge_ratio * self.size)

        # If there is an open trade, wait until the zscore crosses zero
        elif self.position.size > 0 and self.zscore > 0:
            self.log("CLOSE LONG SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
            self.log("Z-SCORE: {}".format(self.zscore))
            self.order_target_size(self.datas[0], 0)
            self.order_target_size(self.datas[1], 0)

        elif self.position.size < 0 and self.zscore < 0:
            self.log("CLOSE SHORT SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
            self.log("Z-SCORE: {}".format(self.zscore))
            self.order_target_size(self.datas[0], 0)
            self.order_target_size(self.datas[1], 0)


#%%
# possible improvement ->  try to stray away from thinking in terms of entry/exit
# and instead position size as a function of signal strength
# add win rates
# 1% Portfolio value rule for position size

# strategie improvment like shorter term cointegration -> look for pair with cointegration on short time frames
# trade shorter time frame and change pair if there is any signal that cointegration starts slowing

from backtrader.analyzers import tradeanalyzer

if __name__ == "__main__":

    days = 300
    std_dev = 1
    size = 10000

    cerebro = bt.Cerebro()

    # Fetch data and find cointegrated pairs
    client = Client(api_key, api_secret)
    data = fetch_crypto_data(20, days, client)
    pairs = find_cointegrated_pairs_2(data)

    # Choose the pair with the smallest p-value
    tickers_pairs = pairs.iloc[0, 0:2]
    window = int(pairs.iloc[0,3])
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
    print("Number of trades: {}".format(trade_analyzer.total.closed))
    print("Half Life: " + str(window) + " hours")
    print("Winning Trades:", results[0].analyzers.trade_analyzer.get_analysis()['won'])
    print("Losing Trades:", results[0].analyzers.trade_analyzer.get_analysis()['lost'])

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

    # performance analytics
    portfolio_stats = strategy_instance.analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)
    quantstats.reports.html(returns, output='stats.html', title='BTC Sentiment')



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

                #delta = np.log(S1/S2)
                #beta = np.polyfit(S1, delta, 1)[0]
                #half_life = np.log(2) / beta

                #res = sm.OLS(S1, sm.add_constant(S2)).fit()
                #half_life_2 = -np.log(2) / res.params[1]

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



#%%
