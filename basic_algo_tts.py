#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:06:08 2022

@author: marcopegoraro
"""

import pandas as pd 
import numpy as np
import matplotlib as plt
import talib

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from api_key_secret import api_key, api_secret
from time import sleep
from binance import ThreadedWebsocketManager

#https://medium.com/geekculture/building-a-basic-crypto-trading-bot-in-python-4f272693c375
#STEP 1: fetch data


# STEP 1: FETCH THE DATA
def fetch_data(ticker, days):
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
    
    historical = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1MINUTE, str(days)+" day ago UTC")
    hist_df = pd.DataFrame(historical)
    
    hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
                        'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
    
    hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')
    hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time']/1000, unit='s')
    
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']
    
    hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)


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

'''
# STEP 3: EXECUTE THE TRADE
def execute_trade(trade_rec_type, trading_ticker):
    global wx_client, HOLDING_QUANTITY
    order_placed = False
    side_value = 'buy' if (trade_rec_type == "BUY") else 'sell'
    try:
        ticker_price_response = wx_client.send("ticker", { "symbol": trading_ticker})
        if (ticker_price_response[0] in [200, 201]):
            current_price = float(ticker_price_response[1]['lastPrice'])
            scrip_quantity = round(INVESTMENT_AMOUNT_DOLLARS/current_price,5) if trade_rec_type == "BUY" else HOLDING_QUANTITY
            
            print(f"PLACING ORDER {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: "
                  f"{trading_ticker}, {side_value}, {current_price}, {scrip_quantity}, {int(time.time() * 1000)} ")
            order_response = wx_client.send("create_order",
                                         {"symbol": trading_ticker, "side": side_value, "type": "limit",
                                          "price": current_price, "quantity": scrip_quantity,
                                          "recvWindow": 10000, "timestamp": int(time.time() * 1000)})
            print(f'ORDER PLACED. RESPONSE: {order_response}')
            
            HOLDING_QUANTITY = scrip_quantity if trade_rec_type == "BUY" else HOLDING_QUANTITY
            order_placed = True
    except:
        print(f"\nALERT!!! UNABLE TO COMPLETE THE ORDER.")

    return order_placed

# all together
def run_bot_for_ticker(ccxt_ticker, trading_ticker):

    currently_holding = False
    while 1:
        # STEP 1: FETCH THE DATA
        ticker_data = fetch_data(ccxt_ticker)
        if ticker_data is not None:
            # STEP 2: COMPUTE THE TECHNICAL INDICATORS & APPLY THE TRADING STRATEGY
            trade_rec_type = get_trade_recommendation(ticker_data)
            print(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}  TRADING RECOMMENDATION: {trade_rec_type}')

            # STEP 3: EXECUTE THE TRADE
            if (trade_rec_type == 'BUY' and not currently_holding) or \
                (trade_rec_type == 'SELL' and currently_holding):
                print(f'Placing {trade_rec_type} order')
                trade_successful = execute_trade(trade_rec_type,trading_ticker)
                currently_holding = not currently_holding if trade_successful else currently_holding

            # SLEEP BEFORE REPEATING THE STEPS
            time.sleep(CANDLE_DURATION_IN_MIN*60)
        else:
            print(f'Unable to fetch ticker data - {ccxt_ticker}. Retrying!!')
            time.sleep(5)
            
'''

if __name__ == "__main__":


    client = Client(api_key, api_secret)
    
    tickers = client.get_all_tickers()
    
    ticker = "ETHBUSD"
    days = '3'
    
    df_hist = fetch_data(ticker, days)
    trade_rec_type = get_trade_recommendation(df_hist)



