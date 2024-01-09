from binance.client import Client
from datetime import datetime, timedelta
import time
import pandas as pd
import matplotlib.pyplot as plt


api_key = 'zKSX5AS3BxDbOASdI9hnXFdLhIFR52aug52oVLFxn4yelDu2CyJKmXztvolrysOZ'

api_secret = 'YYo9kaXiGttMdCeZwWxSQymhv8hnLsYPUiifPeapdM1t44ASdpot4fDR6ioJnz2h'

client = Client(api_key, api_secret)

symbol = 'ETHUSDT'
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