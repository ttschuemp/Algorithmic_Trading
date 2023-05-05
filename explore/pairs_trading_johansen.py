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
        self.hedge_ratio_history = []
        self.spread_history = []



    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print("{} {}".format(dt.isoformat(), txt))


    def calc_hedge_ratio(self, data_a, data_b):

        try:

            result = coint_johansen(np.array([self.data_a, self.data_b]).T, det_order=0, k_ar_diff=1)
            self.hedge_ratio = result.evec.T[0][1] / result.evec.T[0][0]
            self.spread = self.data_a[0] - (self.hedge_ratio * self.data_b[0])
            self.hedge_ratio_history.append(self.hedge_ratio)
            self.spread_history.append(self.spread)
            return(self.spread)

        except Exception as e:

            print(f"Error calculating hedge ratio: {e}")
            return None


        print(self.spread)





    def start(self):
        self.equity = self.broker.get_cash()


    def next(self):

        #if len(self.data_a) >= self.window and len(self.data_b) >= self.window:

        self.equity = self.broker.get_value()
        #self.trade_size = self.equity * self.params.size / self.data_a[0]


        self.spread = self.calc_hedge_ratio(self.data_a, self.data_b)

        #self.log("Hedge ratio:".format(hedge_ratio))







#%%

# why are the winning and losing trades always the same amount?
# ADF with trades most of time pvalue >0.05 -> non stationary. means spread we trade is not stationary

if __name__ == "__main__":
    days = 90
    cerebro = bt.Cerebro()

    # Fetch data and find cointegrated pairs
    client = Client(api_key, api_secret)
    data = fetch_crypto_data(30, days, client)
    pairs = find_cointegrated_pairs_hurst(data)

    window = int(pairs['Half Life'][0])
    #window = 5000
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

    strategy_instance = results[0]

    plt.plot(strategy_instance.hedge_ratio_history)
    plt.title(f'Spread {list(tickers_pairs)}')

    plt.plot(strategy_instance.spread_history)
    plt.title(f'Spread {list(tickers_pairs)}')




#%%
