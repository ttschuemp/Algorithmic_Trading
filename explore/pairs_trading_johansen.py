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
from statsmodels.tsa.stattools import adfuller
import backtrader.broker as btbroker
from src.load_data import fetch_crypto_data, fetch_data
from src.pairs_trading_functions import*
from binance import Client
from src.api_key_secret import api_key, api_secret, path_zert

from statsmodels.tsa.vector_ar.vecm import coint_johansen


class PairsTrading(bt.Strategy):
    params = (
        ("window", None),
        ("std_dev", None),
        ("size", None)
    )

    def __init__(self):
        self.data_a = self.datas[0].close
        self.data_b = self.datas[1].close
        self.window = self.params.window
        self.hedge_ratio = None
        self.spread = None
        self.zscore = None
        self.upper_bound = self.params.std_dev
        self.lower_bound = -self.params.std_dev
        self.size = self.params.size
        self.trade_size = 0

        self.hedge_ratio_history = []
        self.spread_history = []
        self.zscore_history = []



    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print("{} {}".format(dt.isoformat(), txt))


    def calc_hedge_ratio(self, data_a, data_b):

        try:

            result = coint_johansen(np.array([data_a, data_b]).T, det_order=0, k_ar_diff=1)
            self.hedge_ratio = result.evec.T[0][1] / result.evec.T[0][0]
            self.spread = data_a[-1] - (self.hedge_ratio * data_b[-1])
            self.hedge_ratio_history.append(self.hedge_ratio)
            self.spread_history.append(self.spread)

            spread_mean = pd.Series(self.spread_history).rolling(self.params.window).mean().iloc[-1]
            spread_std_dev = pd.Series(self.spread_history).rolling(self.params.window).std().iloc[-1]
            self.zscore = (self.spread - spread_mean) / spread_std_dev
            self.zscore_history.append(self.zscore)

            return self.zscore

        except Exception as e:

            print(f"Error calculating hedge ratio: {e}")
            return None



    def start(self):
        self.equity = self.broker.get_cash()


    def next(self):

        if len(self.data_a) >= self.window and len(self.data_b) >= self.window:

            self.equity = self.broker.get_value()
            self.trade_size = self.equity * self.params.size / self.data_a[0]

            data_a = self.data_a.get(size=self.window)
            data_b = self.data_b.get(size=self.window)
            self.zscore = self.calc_hedge_ratio(data_a, data_b)

            # Check if there is already an open trade
            if self.getposition().size == 0:
                if (self.zscore < self.lower_bound):
                    # Buy the spread
                    self.log("BUY SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                    self.log("Z-SCORE: {}".format(self.zscore))
                    self.log("Portfolio Value: {}".format(self.equity))
                    #self.log("Hurst: {}".format(self.hurst_exponent))
                    #self.log("Hurst 2: {}".format(self.hurst_exponent_2))
                    #self.log("ADF P-Value: {}".format(adfuller(self.spread_history)[1]))
                    self.order_target_size(self.datas[0], self.trade_size)
                    self.order_target_size(self.datas[1], -self.hedge_ratio * self.trade_size)

                elif (self.zscore > self.upper_bound):
                    # Sell the spread
                    self.log("SELL SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                    self.log("Z-SCORE: {}".format(self.zscore))
                    #self.log("Portfolio Value: {}".format(self.equity))
                    #self.log("Hurst: {}".format(self.hurst_exponent))
                    #self.log("Hurst 2: {}".format(self.hurst_exponent_2))
                    #self.log("ADF P-Value: {}".format(adfuller(self.spread_history)[1]))
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

# why are the winning and losing trades always the same amount?
# ADF with trades most of time pvalue >0.05 -> non stationary. means spread we trade is not stationary

if __name__ == "__main__":
    days = 180
    cerebro = bt.Cerebro()

    # Fetch data and find cointegrated pairs
    client = Client(api_key, api_secret)
    data = fetch_crypto_data(30, days, client)
    pairs = find_cointegrated_pairs_hurst(data)

    #window = int(pairs['Half Life'][0])
    window = 1000
    std_dev = 1
    size = 0.02

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

    final_value = cerebro.broker.getvalue()
    print("Final portfolio value: ${}".format(final_value))

    # Get the analyzers and print the results
    trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
    print("Starting cash: ${}".format(cerebro.broker.startingcash))
    print("Ending cash: ${}".format(cerebro.broker.getvalue()))
    print("Total return: {:.2f}%".format(100*((cerebro.broker.getvalue()/cerebro.broker.startingcash)-1)))
    print("Half-Live: " + str(window) + " hours")
    print("Number of trades: {}".format(trade_analyzer.total.closed))
    print("Winning Trades:", results[0].analyzers.trade_analyzer.get_analysis()['won']['total'])
    print("Losing Trades:", results[0].analyzers.trade_analyzer.get_analysis()['lost']['total'])
    print("Win Ratio:", results[0].analyzers.trade_analyzer.get_analysis()['won']['total'] /
          trade_analyzer.total.closed)


    strategy_instance = results[0]

    # Plot the spread, zscore, and hedge ratio
    plt.subplot(4, 1, 1)
    plt.plot(strategy_instance.spread_history)
    plt.title(f'Spread {list(tickers_pairs)}')

    plt.subplot(4, 1, 2)
    plt.plot(strategy_instance.zscore_history)
    plt.axhline(strategy_instance.upper_bound, color='r')
    plt.axhline(strategy_instance.lower_bound, color='r')
    plt.title("Z-score")
    plt.legend(["Z-score"])

    plt.subplot(4, 1, 3)
    plt.plot(strategy_instance.hedge_ratio_history)
    plt.title("Hedge ratio")
    plt.legend(["Hedge ratio"])

    plt.tight_layout()
    plt.savefig('strategy')
    plt.show()

    # create cerebro chart
    cerebro.plot(iplot=True, volume=False)

    # create quantstats charts & statistics html
    portfolio_stats = strategy_instance.analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)

    quantstats.reports.html(returns, output='stats.html', title='Backtrade Pairs')



#%%
