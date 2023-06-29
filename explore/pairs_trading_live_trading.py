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
from src.api_key_secret import*


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
        #if len(self.spread_history) >= self.params.window:


         #  self.hurst_exponent, c, data_hist = compute_Hc(self.spread_history, kind='change', simplified=False)

         #  hurst_hist, c, data_hist = compute_Hc(self.spread_history_full, kind='change', simplified=False)
         #   self.hurst_history_2.append(hurst_hist)


    def start(self):
        self.equity = self.broker.get_cash()


    def next(self):
        # the trade_size part is not correct
        self.equity = self.broker.get_value()
        self.trade_size = float(account_info['balances'][0]['free']) * self.params.size
        self.calc_hedge_ratio()

        # Check if there is already an open trade
        if self.getposition().size == 0:
            # if (self.zscore < self.lower_bound) and (0 < self.hurst_exponent < 1):
            if (self.zscore < self.lower_bound):
                # Buy the spread
                self.log("BUY SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                self.log("Z-SCORE: {}".format(self.zscore))
                self.log("Portfolio Value: {}".format(self.equity))
                #self.log("Hurst: {}".format(self.hurst_exponent))
                #self.log("Hurst 2: {}".format(self.hurst_exponent_2))
                self.log("ADF P-Value: {}".format(adfuller(self.spread_history)[1]))

                client.create_order(symbol='BTCBUSD', side = 'BUY', type = 'MARKET', quantity = self.trade_size)
                client.create_order(symbol='ETHBUSD', side = 'SELL', type = 'MARKET', quantity = self.hedge_ratio * self.trade_size )

            #elif (self.zscore > self.upper_bound) and (0 < self.hurst_exponent < 1):
            elif (self.zscore > self.upper_bound):
                # Sell the spread
                self.log("SELL SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                self.log("Z-SCORE: {}".format(self.zscore))
                self.log("Portfolio Value: {}".format(self.equity))
                #self.log("Hurst: {}".format(self.hurst_exponent))
                #self.log("Hurst 2: {}".format(self.hurst_exponent_2))
                self.log("ADF P-Value: {}".format(adfuller(self.spread_history)[1]))

                client.create_order(symbol='BTCBUSD', side = 'SELL', type = 'MARKET', quantity = self.trade_size)
                client.create_order(symbol='ETHBUSD', side = 'SELL', type = 'MARKET', quantity = self.hedge_ratio * self.trade_size)


        # If there is an open trade, wait until the zscore crosses zero
        elif self.getposition().size > 0 and self.zscore > 0:
            self.log("CLOSE LONG SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
            self.log("Z-SCORE: {}".format(self.zscore))
            self.log("Portfolio Value: {}".format(self.equity))

            client.create_order(symbol='BTCBUSD', side = 'SELL', type = 'MARKET', quantity = self.trade_size)
            client.create_order(symbol='ETHBUSD', side = 'SELL', type = 'MARKET', quantity = self.hedge_ratio * self.trade_size)


        elif self.getposition().size < 0 and self.zscore < 0:
            self.log("CLOSE SHORT SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
            self.log("Z-SCORE: {}".format(self.zscore))
            self.log("Portfolio Value: {}".format(self.equity))
            client.create_order(symbol='BTCBUSD', side = 'SELL', type = 'MARKET', quantity = self.trade_size)
            client.create_order(symbol='ETHBUSD', side = 'SELL', type = 'MARKET', quantity = self.hedge_ratio * self.trade_size)


#%%
if __name__ == "__main__":
    days = 13
    cerebro = bt.Cerebro()

    # Fetch data and find cointegrated pairs
    client = Client(api_key_testnet, api_secret_testnet, testnet=True)
    client.API_URL = 'https://testnet.binance.vision/api'
    data = fetch_crypto_data(50, days, client)
    pairs = find_cointegrated_pairs_hurst(data)

    window = int(pairs['Half Life'][0])
    window = 150
    std_dev = 1
    size = 0.02

    # Choose the pair with the smallest p-value
    tickers_pairs = pairs.iloc[0, 0:2]
    #tickers_pairs = ['BTCBUSD', 'ETHBUSD']
    print(f'trading pair: ' + str(tickers_pairs))

    pair_1 = symbol_string_conversion(tickers_pairs[0], 'BUSD')
    pair_2 = symbol_string_conversion(tickers_pairs[1], 'BUSD')

    my_exchange = 'Binance' # example of crypto exchange
    method_to_call = getattr(ccxt,my_exchange.lower()) # retrieving the method #from ccxt whose name matches the given exchange name
    exchange_obj = method_to_call() # defining an exchange object

    pair1_price_data = exchange_obj.fetch_ticker(pair_1)
    closing_price = pair_price_data['close']



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

    # Print the final portfolio value
    final_value = cerebro.broker.getvalue()
    print("Final portfolio value: ${}".format(final_value))

#%%
from binance import Client
#from src.api_key_secret import api_key_testnet, api_secret_testnet, path_zert

api_key_testnet = '6FVEREC1YOenBUKfisdPXaJHNn8kM3lzxWCgFRDUvyY9fKM2H17pZz6wNg2Spf71'

api_secret_testnet = '0wlHotYWjFIvpZc69FnESbfRhaDUMtcidNJX72obsocGRlH9Feg90rVkT7YCKUqg'

client = Client(api_key_testnet, api_secret_testnet, testnet=True, verfiy: )
client.API_URL = 'https://testnet.binance.vision/api'

account_info = client.get_account()
trade_size = float(account_info['balances'][0]['free']) * 0.02


symbol = 'BTCBUSD'
quantity = 0.01

buy_order = client.create_order(symbol=tickers_pairs[0], side = 'BUY', type = 'MARKET', quantity=round(trade_size / 30000, 6))

sell_order = client.create_order(symbol=symbol, side = 'SELL', type = 'MARKET', quantity=quantity)

#%%
import datetime

symbol = 'BTCBUSD'

days = 50
end_time = datetime.datetime.now()
start_time = end_time - datetime.timedelta(days=days)

klines = client.get_historical_klines(symbol, client.KLINE_INTERVAL_1HOUR, start_time.strftime("%d %b %Y %H:%M:%S"), end_time.strftime("%d %b %Y %H:%M:%S"))
df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

#%%
client2 = Client(api_key_testnet, api_secret_testnet)
klines2 = client2.get_historical_klines(symbol, client.KLINE_INTERVAL_1HOUR, start_time.strftime("%d %b %Y %H:%M:%S"), end_time.strftime("%d %b %Y %H:%M:%S"))
df2 = pd.DataFrame(klines2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')





#%%
client.create_order(symbol='BTCBUSD', side = 'BUY', type = 'MARKET', quantity=round(trade_size / 30000, 6))
client.create_order(symbol='ETHBUSD', side = 'SELL', type = 'MARKET', quantity=round(trade_size / 30000, 6))

# buy dollar amount
client.create_order(symbol='ETHBUSD', side = 'BUY', type = 'MARKET', quoteOrderQty=15)
client.create_order(symbol='ETHBUSD', side = 'BUY', type = 'MARKET', quoteOrderQty=(hedge_ratio * trade_size))
#%%
# ccxt
import ccxt

my_exchange = 'Binance' # example of crypto exchange
method_to_call = getattr(ccxt,my_exchange.lower()) # retrieving the method #from ccxt whose name matches the given exchange name
exchange_obj = method_to_call() # defining an exchange object

ticker = 'BTC/BUSD'
pair_price_data = exchange_obj.fetch_ticker(ticker)
data1 = pd.DataFrame(pair_price_data)
data1['timestamp'] = pd.to_datetime(data1['timestamp'], unit='ms')
data1.set_index('timestamp', inplace=True)


#%%
from datetime import datetime, timedelta
import time


day = 15
hours = days * 24


def symbol_string_conversion(tickers_pairs, stable_coin):
    index = tickers_pairs.index(stable_coin)  # Find the index of "BUSD"
    symbol = tickers_pairs[:index] + '/' + tickers_pairs[index:]
    return symbol


symbol = symbol_string_conversion(tickers_pairs[1], 'BUSD')
exchange = 'binance'

hist_start_date = datetime.utcnow() - timedelta(hours=hours)
data = bt.feeds.CCXT(exchange=exchange,
                     symbol=symbol,
                     timeframe=bt.TimeFrame.Hours,
                     fromdate=hist_start_date,
                     ohlcv_limit=444)
#%%
