#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:19:05 2022

@author: marcopegoraro
https://www.youtube.com/watch?v=X50-c54BWV8
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import talib
import ta

from binance import Client, AsyncClient, ThreadedWebsocketManager, ThreadedDepthCacheManager
from api_key_secret import*
import time
from binance import ThreadedWebsocketManager

#https://medium.com/geekculture/building-a-basic-crypto-trading-bot-in-python-4f272693c375
#STEP 1: fetch data

client = Client(api_key_testnet, api_secret_testnet, testnet=True)
client.API_URL = 'https://testnet.binance.vision/api'

account_info = client.get_account()

# STEP 1: FETCH THE DATA
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
    
def trading_technicals(df):
    df['%K'] = ta.momentum.stoch(df.High,df.Low,df.Close, window=14, smooth_window=3)
    df['%D'] = df['%K'].rolling(3).mean()
    df['rsi'] = ta.momentum.rsi(df.Close, window=14)
    df['macd'] = ta.trend.macd_diff(df.Close)
    df.dropna(inplace=True)
    
    
trading_technicals(df)
    
class Signals:
    def __init__(self,df, lags):
        self.df = df
        self.lags = lags
        
        
    def gettrigger(self):
        dfx = pd.DataFrame()
        for i in range(self.lags + 1):
            mask = (self.df['%K'].shift(i)<20) & (self.df['%D'].shift(i)<20)
            dfx = dfx.append(mask, ignore_index=True)
        return dfx.sum(axis=0)
    
    
    def decide(self):
        self.df['trigger'] = np.where(self.gettrigger(),1,0)
        self.df['Buy'] = np.where((self.df.trigger) &
        (self.df['%K'].between(20,80)) & (self.df['%D'].between(20,80))
                                        & (self.df.rsi > 50) & (self.df.macd >0),1,0)
        
        
inst = Signals(df, 50)   #lag 5 is bether
inst.decide()


def strategy(pair, qty, open_position=False):
    df = fetch_data(pair, '1m', '100')
    trading_technicals(df)
    inst = Signals(df,25)
    inst.decide()
    print(f'current Close ist '+str(df.Close.iloc[-1]))
    if df.Buy.iloc[-1]:
        order = client.create_order(symbol=pair,
                                    side = 'BUY',
                                    type = 'MARKET',
                                    quantity=qty)
        print(order)
        buyprice = float(order['fills'][0]['price'])
        open_position = True
    while open_position:
        time.sleep(0.5)
        df = fetch_data(pair, '1m', '2')
        print(f'current Close '+ str(df.Close.iloc[-1]))
        print(f'current Target '+ str(buyprice * 1.005))
        print(f'current Stop is ' + str(buyprice * 0.995))
        if df.Close[-1] <= buyprice*0.995 or df.Close[-1] >= 1.005* buyprice:
            order = client.create_order(symbol=pair,
                                        side = 'SELL',
                                        type = 'MARKET',
                                        quantity=qty)
            print(order)
            break
        
        
#loop for whole day
while True:
    strategy('BNBUSDT',1)
    time.sleep(0.5)




#%%
