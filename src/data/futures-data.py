from binance.client import Client
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import nested_lookup as nl
from pybit.unified_trading import HTTP

# binance
api_key = 'zKSX5AS3BxDbOASdI9hnXFdLhIFR52aug52oVLFxn4yelDu2CyJKmXztvolrysOZ'
api_secret = 'YYo9kaXiGttMdCeZwWxSQymhv8hnLsYPUiifPeapdM1t44ASdpot4fDR6ioJnz2h'

# bybit

api_key_by = 'kezeQXNipK0hy06iAd'
api_secret_by = 'DD8F7b8UykIfnDYJ619UMBZaailhtQlWECw2'


client = Client(api_key, api_secret)

symbol = 'BTCUSDT'
interval = '1d'
limit = 1000

# fetch spot
klines_spot = client.get_klines(
    symbol=symbol,
    interval=interval,
    limit=limit

)
df_spot = pd.DataFrame(klines_spot, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
# Convert timestamp to readable date
df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'], unit='ms')
spot_prices_close = df_spot['open'].astype(float)


r = client.futures_continous_klines(
    pair=symbol,
    contractType='CURRENT_QUARTER',
    interval=interval,
    limit=limit
)
df_future_next_quarter = pd.DataFrame(r, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
# Convert timestamp to readable date
df_future_next_quarter['timestamp'] = pd.to_datetime(df_future_next_quarter['timestamp'], unit='ms')
future_next_quarter_close = df_future_next_quarter['open'].astype(float)


plt.plot(pd.DataFrame(future_next_quarter_close/spot_prices_close).set_index(df_spot['timestamp']), color='blue')
plt.axhline(1, color='r')
plt.legend(["Spread " + f'{symbol}'])
plt.rcParams.update({'font.size': 18})
plt.show()

# bybit api

session = HTTP(testnet=True)

by = session.get_kline(
        category="spot",
        symbol="BTCUSDT",
        interval='1',
        limit=1000
    )

import requests

# Bybit API base URL
base_url = 'https://api.bybit.com'

# Set the endpoint for tickers data
endpoint = '/v5/market/kline'

# Symbol for the futures trading pair (e.g., BTCUSD)
symbol = 'ETH-23FEB24'

# Construct the request parameters
params = {'category': 'linear',
          'symbol': symbol,
          'interval': 'D',
          'limit': 200}

# Make the API request
response = requests.get(base_url + endpoint, params=params)
data = response.json()

prices = nl.nested_lookup('list', data)

flat_data = [item for sublist in prices[0] for item in sublist]

df = pd.DataFrame(
    [flat_data[i:i+7] for i in range(0, len(flat_data), 7)],
    columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'turnover']
)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')


plt.plot(df['Close'].set_index(df['Timestamp']), color='blue')
plt.legend(["Spread " + f'{symbol}'])
plt.rcParams.update({'font.size': 18})
plt.show()


# HIST FUNDING RATES

from datetime import datetime
from pybit.unified_trading import HTTP

session = HTTP(
    testnet=False,
    api_key=api_key,
    api_secret=api_secret,
)

# Assuming you have already created an instance of MarketHTTP with the appropriate API credentials


# Set the required parameters
category = 'linear'
limit = 1000
symbol = 'ETHPERP'  # Replace with the desired symbol

# Additional optional parameters
start_time = int(datetime.timestamp(datetime(2023, 1, 1)))  # Replace with the desired start time in Unix timestamp format
end_time = int(datetime.timestamp(datetime(2023, 12, 31)))  # Replace with the desired end time in Unix timestamp format

# Make the API call to get historical funding rates
funding_rate_history = session.get_funding_rate_history(category=category, symbol=symbol, limit=limit)

# Print or process the response as needed
print(funding_rate_history)
