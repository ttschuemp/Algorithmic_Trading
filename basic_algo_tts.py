#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:06:08 2022

@author: marcopegoraro
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import talib
import ta

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from api_key_secret import api_key, api_secret
from time import sleep
from binance import ThreadedWebsocketManager


#https://medium.com/geekculture/building-a-basic-crypto-trading-bot-in-python-4f272693c375
#STEP 1: fetch data


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

# STEP 2: COMPUTE THE TECHNICAL INDICATORS & APPLY THE TRADING STRATEGY
def get_trade_recommendation(df_hist):
  
    macd_result, final_result = 'WAIT','WAIT'
    # BUY or SELL based on MACD crossover points and the RSI value at that point
    macd, signal, hist = talib.MACD(df_hist['Close'], fastperiod = 12, slowperiod = 26, signalperiod = 9)
    last_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    if not np.isnan(prev_hist) and not np.isnan(last_hist):
        # If hist value has changed from negative to positive or vice versa, it indicates a crossover
        macd_crossover = (abs(last_hist + prev_hist)) != (abs(last_hist) + abs(prev_hist))
        if macd_crossover:
            macd_result = 'BUY' if last_hist > 0 else 'SELL'
            
    if macd_result != 'WAIT':
        rsi = talib.RSI(df_hist['close'], timeperiod = 14)
        last_rsi = rsi.iloc[-1]
        if (last_rsi <= RSI_OVERSOLD):
            final_result = 'BUY'
        elif (last_rsi >= RSI_OVERBOUGHT):
            final_result = 'SELL'
    return final_result

    
def trading_technicals(df):
    df['%K'] = ta.momentum.stoch(df.High,df.Low,df.Close, window=14, smooth_window=3)
    df['%D'] = df['%K'].rolling(3).mean()
    df['rsi'] = ta.momentum.rsi(df.Close, window=14)
    df['macd'] = ta.trend.macd_diff(df.Close)
    df.dropna(inplace=True)

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




if __name__ == "__main__":


    client = Client(api_key, api_secret)
    
    tickers = client.get_all_tickers()
    
    ticker = "BTCUSDT"
    interval = '1m'
    lookback = '300'
    
    df_hist = fetch_data(ticker, interval, lookback)
    trading_technicals(df_hist)
    
        
    inst = Signals(df_hist, 5)   #lag 5 is bether
    inst.decide()

    figure, axis = plt.subplots(2, 2)
      
    axis[0, 0].plot(df_hist['Open'])
    axis[0, 0].set_title(df_hist['Open'].name)
    axis[0, 0].plot(df_hist.Open[df_hist['Buy']==1], "s")
    
    axis[0, 1].plot(df_hist['%K'])
    axis[0, 1].set_title(df_hist['%K'].name)
      
    axis[1, 0].plot(df_hist['%D'])
    axis[1, 0].set_title(df_hist['%D'].name)
      
    axis[1, 1].plot(df_hist['rsi'])
    axis[1, 1].set_title(df_hist['rsi'].name)
      
    # Combine all the operations and display
    plt.show()
    

    
    
    
    #trade_rec_type = get_trade_recommendation(df_hist)
    
    
    
    
    
    



