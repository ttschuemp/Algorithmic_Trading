import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import talib
import ta
from datetime import datetime
import backtrader as bt
from binance import Client, AsyncClient, ThreadedWebsocketManager, ThreadedDepthCacheManager
from api_key_secret import api_key, api_secret
import time
from binance import ThreadedWebsocketManager

client = Client(api_key,api_secret)

# Create a data feed
def fetch_data(ticker, interval, lookback):
    '''
         1499040000000,      // Open time
         "0.01634790",       // Open
         "0.80000000",       // High
         "0.01575800",       // Low
         "0.01577100",       // Close
         "148976.11427815",  // Volume
         1499644799999,      // Close time
         "2434.19055334",    // Quote asset volume
         308,                // Number of trades
         "1756.87402397",    // Taker buy base asset volume
         "28.46694368",      // Taker buy quote asset volumeic#     "17928899.62484339" // Ignore.
    '''
    
    hist_df = pd.DataFrame(client.get_historical_klines(ticker, interval, lookback + 'min ago UTC'))
    hist_df = hist_df.iloc[:,:6]
    hist_df.columns = ['Time','Open', 'High', 'Low', 'Close', 'Volume']
    hist_df = hist_df.set_index('Time')
    hist_df.index = pd.to_datetime(hist_df.index, unit='ms')
    hist_df = hist_df.astype(float)
    return hist_df


data = fetch_data("BTCUSDC", '1m', '5000')

cerebro = bt.Cerebro() 

# Create a subclass of Strategy to define the indicators and logic

class SmaCross(bt.Strategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=100)  # fast moving average
        sma2 = bt.ind.SMA(period=200)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market 
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position

feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(feed)
cerebro.addstrategy(SmaCross)  # Add the trading strategy
cerebro.addsizer(bt.sizers.PercentSizer, percents=50)
cerebro.run()  # run it all
cerebro.plot(iplot=False) # create a "Cerebro" engine instance


