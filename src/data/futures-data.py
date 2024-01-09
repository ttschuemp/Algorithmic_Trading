from binance.client import Client
from datetime import datetime, timedelta
import time
import pandas as pd
import matplotlib.pyplot as plt


api_key = 'zKSX5AS3BxDbOASdI9hnXFdLhIFR52aug52oVLFxn4yelDu2CyJKmXztvolrysOZ'

api_secret = 'YYo9kaXiGttMdCeZwWxSQymhv8hnLsYPUiifPeapdM1t44ASdpot4fDR6ioJnz2h'

client = Client(api_key, api_secret)

symbol = 'BTCUSDT'
interval = '1d'
limit = 1000

# fetch futures
klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
df2 = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')
future_prices_close = df2['close'].astype(float)


# fetch spot
klines_spot = client.get_klines(symbol=symbol, interval=interval, limit=limit)
df_spot = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
# Convert timestamp to readable date
df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'], unit='ms')
spot_prices_close = df_spot['close'].astype(float)

plt.plot(future_prices_close)
plt.savefig('future price')
