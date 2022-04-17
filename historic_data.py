#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:57:32 2022

@author: tobiastschuemperlin
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from api_key_secret import api_key, api_secret
from time import sleep
from binance import ThreadedWebsocketManager


client = Client(api_key, api_secret)

tickers = client.get_all_tickers()


#  [
#   [
#     1499040000000,      // Open time
#     "0.01634790",       // Open
#     "0.80000000",       // High
#     "0.01575800",       // Low
#     "0.01577100",       // Close
#     "148976.11427815",  // Volume
#     1499644799999,      // Close time
#     "2434.19055334",    // Quote asset volume
#     308,                // Number of trades
#     "1756.87402397",    // Taker buy base asset volume
#     "28.46694368",      // Taker buy quote asset volumeic#     "17928899.62484339" // Ignore.
#   ]
# ]



historical = client.get_historical_klines("ETHBUSD", Client.KLINE_INTERVAL_1MINUTE, "3 day ago UTC")
hist_df = pd.DataFrame(historical)

hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
                    'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')
hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time']/1000, unit='s')

numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']

hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)



hist_df['Open'].plot()



