import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
#from api_key_secret import api_key, api_secret

#client = Client(api_key, api_secret)
#client.API_URL = 'https://testnet.binance.vision/api'

# make a function to load data
def load_data(path):
    # load csv file in data pairs
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    # rename column index to date
    data.index.name = 'date'
    return data

def fetch_data(ticker, interval, lookback):
    # fetch binance data
    hist_df = pd.DataFrame(client.get_historical_klines(ticker, interval, lookback + 'hours ago UTC'))
    hist_df = hist_df.iloc[:,:6]
    hist_df.columns = ['Time','Open', 'High', 'Low', 'Close', 'Volume']
    hist_df = hist_df.set_index('Time')
    hist_df.index = pd.to_datetime(hist_df.index, unit='ms')
    hist_df = hist_df.astype(float)
    return hist_df

def fetch_crypto_data(top_n, days, client):
    # Get the top N cryptocurrencies by market cap from CoinGecko API
    response = requests.get(f"https://api.coingecko.com/api/v3/coins/"
                            f"markets?vs_currency=usd&order=market_cap_desc&per_page={top_n}&"
                            f"page=1&sparkline=false&price_change_percentage=24h"
                            #, verify="C:\DevLab\Zscaler Zertifikat.cer"
                            )
    top_cryptos = response.json()

    # Construct a list of ticker pairs for the top cryptocurrencies against USDT
    ticker_pairs = [crypto["symbol"].lower() + "usdt" for crypto in top_cryptos]

    ticker_pairs = [f"{ticker.upper()}" for ticker in ticker_pairs]

    # Define the start and end time for the historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    # Loop over the trading pairs and retrieve the historical intra-day 1-hour ticks
    df_list = []
    for symbol in ticker_pairs:
        try:
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, start_time.strftime("%d %b %Y %H:%M:%S"), end_time.strftime("%d %b %Y %H:%M:%S"))
            if len(klines) >= days*24-1:
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.drop(['open', 'high', 'low', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)
                df.columns = [symbol]
                df_list.append(df)
                print(f"Retrieved {len(klines)} klines for {symbol}")
            else:
                print(f"Skipping {symbol} due to insufficient data points (has {len(klines)} klines)")
        except Exception as e:
            print(f"Error retrieving data for {symbol}: {e}")

        # Throttle the API requests to avoid hitting the rate limit
        time.sleep(1)

    # Combine the dataframes into one

    return pd.concat(df_list, axis=1)

