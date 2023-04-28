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

        #self.ols_slope = btind.OLS_Slope_InterceptN(self.data_a, self.data_b, period=self.params.window)
        self.hedge_ratio = None
        self.hurst_exponent = None
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print("{} {}".format(dt.isoformat(), txt))


    def calc_hedge_ratio(self):

            if len(self.data_a) < self.window:
                return
            else:
                a = self.data_a[-self.window:]
                b = self.data_b[-self.window:]

                result = coint_johansen(np.array([a,b].T), det_order=0, k_ar_diff=1)
                print(result.eigvec[0][1])
"""
            result = coint_johansen(np.array([rolling_x.mean(), rolling_y.mean()]).T, det_order=0, k_ar_diff=1)
            johansen_hedge_ratio = result.evec[0, 1] / result.evec[0, 0]
            print(johansen_hedge_ratio)
            self.hedge_ratio = johansen_hedge_ratio
            print(self.hedge_ratio)
            #hedge_ratio = self.ols_slope.slope[0]
            spread = self.data_a[0] - (self.hedge_ratio * self.data_b[0])
            self.spread_history.append(spread)
            spread_mean = pd.Series(self.spread_history).rolling(self.params.window).mean().iloc[-1]
            spread_std_dev = pd.Series(self.spread_history).rolling(self.params.window).std().iloc[-1]
            self.zscore = (spread - spread_mean) / spread_std_dev

            self.spread_history_full.append(spread)
            self.zscore_history.append(self.zscore)
            self.hedge_ratio_history.append(hedge_ratio)

            # calc hurst exponent
            if len(self.spread_history) >= self.params.window:

                #if adfuller(self.spread_history)[1] <= 0.1:
                lags = range(2, self.params.window // 2)

                # Calculate the array of the variances of the lagged differences
                tau = [np.sqrt(np.abs(pd.Series(self.spread_history) - pd.Series(self.spread_history).shift(lag)).dropna().var()) for lag in lags]

                # Use a linear fit to estimate the Hurst Exponent
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                self.hurst_exponent = poly[0] * 2
"""

    def start(self):
        self.equity = self.broker.get_cash()


    def next(self):

        self.equity = self.broker.get_value()
        self.trade_size = self.equity * self.params.size / self.data_a[0]
        self.calc_hedge_ratio()





#%%

# why are the winning and losing trades always the same amount?
# ADF with trades most of time pvalue >0.05 -> non stationary. means spread we trade is not stationary

if __name__ == "__main__":
    days = 90
    cerebro = bt.Cerebro()

    # Fetch data and find cointegrated pairs
    client = Client(api_key, api_secret)
    data = fetch_crypto_data(20, days, client)
    pairs = find_cointegrated_pairs_hurst(data)

    window = int(pairs['Half Life'][0])
    #window = 1000
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
    #slippage = 0.001
    #cerebro.broker = btbroker.BackBroker(slip_perc=slippage)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

    # Run the backtest
    results = cerebro.run()



#%%