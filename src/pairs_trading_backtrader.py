# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import backtrader.indicators as btind
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import pandas as pd
import numpy as np
import collections
from binance import Client
from statsmodels.tsa.stattools import adfuller
from src.api_key_secret import api_key, api_secret #, path_zert



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
        self.hurst_history_1 = []
        self.hurst_history_2 = []

        self.ols_slope = btind.OLS_Slope_InterceptN(self.data_a, self.data_b, period=self.params.window)
        self.hurst_exponent = None
        self.hurst_exponent_2 = None


    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print("{} {}".format(dt.isoformat(), txt))


    def calc_hedge_ratio(self):

        hedge_ratio = self.ols_slope.slope[0]
        spread = self.data_a[0] - (hedge_ratio * self.data_b[0])
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
        
            #lags = range(2, len(self.spread_history) // 2)
        
            # Calculate the array of the variances of the lagged differences
            #tau = [np.sqrt(np.abs((self.spread_history_full) - pd.Series(self.spread_history_full).shift(lag)).dropna().var()) for lag in lags]
        
            # Use a linear fit to estimate the Hurst Exponent
            #poly = np.polyfit(np.log(lags), np.log(tau), 1)
            #self.hurst_exponent = poly[0] * 2
        
            #self.hurst_history_1.append(self.hurst_exponent)

            self.hurst_exponent, c, data_hist = compute_Hc(self.spread_history, kind='change', simplified=False)

            hurst_hist, c, data_hist = compute_Hc(self.spread_history_full, kind='change', simplified=False)
            self.hurst_history_2.append(hurst_hist)


    def start(self):
        self.equity = self.broker.get_cash()


    def next(self):
        # the trade_size part is not correct
        self.equity = self.broker.get_value()
        self.trade_size = self.equity * self.params.size / self.data_a[0]
        self.calc_hedge_ratio()

                # Check if there is already an open trade
                if self.getposition().size == 0:
                    if (self.zscore < self.lower_bound) and (0 < self.hurst_exponent < 1):
                        # Buy the spread
                        self.log("BUY SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                        self.log("Z-SCORE: {}".format(self.zscore))
                        self.log("Portfolio Value: {}".format(self.equity))
                        self.log("Hurst: {}".format(self.hurst_exponent))
                        #self.log("Hurst 2: {}".format(self.hurst_exponent_2))
                        self.log("ADF P-Value: {}".format(adfuller(self.spread_history)[1]))
                        self.order_target_size(self.datas[0], self.trade_size)
                        self.order_target_size(self.datas[1], -self.hedge_ratio * self.trade_size)

                    elif (self.zscore > self.upper_bound) and (0 < self.hurst_exponent < 1):
                        # Sell the spread
                        self.log("SELL SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                        self.log("Z-SCORE: {}".format(self.zscore))
                        self.log("Portfolio Value: {}".format(self.equity))
                        self.log("Hurst: {}".format(self.hurst_exponent))
                        #self.log("Hurst 2: {}".format(self.hurst_exponent_2))
                        self.log("ADF P-Value: {}".format(adfuller(self.spread_history)[1]))
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
