#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:39:12 2022

@author: marcopegoraro
"""
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import talib
import ta
import backtrader as bt
import backtrader.plot
import datetime



from binance import Client, AsyncClient, ThreadedWebsocketManager, ThreadedDepthCacheManager
from api_key_secret import api_key, api_secret
import time
from binance import ThreadedWebsocketManager

#https://medium.com/geekculture/building-a-basic-crypto-trading-bot-in-python-4f272693c375
#STEP 1: fetch data

# matplotlib.use('Qt5Agg')
# plt.switch_backend('Qt5Agg')


client = Client(api_key,api_secret)
client.API_URL = 'https://testnet.binance.vision/api'


cerebro = bt.Cerebro()

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


df = fetch_data("BNBUSDT", '1m', '100')
feed = bt.feeds.PandasData(dataname=df)
cerebro.adddata(feed)
cerebro.run()
cerebro.plot(height= 30, iplot= False)

#plot still doesnt show up
