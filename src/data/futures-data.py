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

params = {
    'category': 'linear',
}

# Make the API request
response = session.get_instruments_info(**params)
all_perps = [instrument['symbol'] for instrument in response['result']['list']]

# Set the required parameters
category = 'linear'
limit = 1000

# Fetch funding rates for the first 10 perpetual contracts
num_perps = 10
plt.figure(figsize=(12, 6))

for symbol in all_perps[:num_perps]:
    # Make the API call to get historical funding rates
    funding_rate_history = session.get_funding_rate_history(category=category, symbol=symbol, limit=limit)

    funding_rates = [float(entry['fundingRate']) for entry in funding_rate_history['result']['list']]
    timestamps = [int(entry['fundingRateTimestamp']) for entry in funding_rate_history['result']['list']]

    # Convert timestamps to readable format (assuming timestamps are in milliseconds)
    readable_timestamps = [datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S') for timestamp in timestamps]

    df = pd.DataFrame({
        'Symbol': [entry['symbol'] for entry in funding_rate_history['result']['list']],
        'Funding Rate': funding_rates,
        'Timestamp': timestamps,
        'Readable Timestamp': readable_timestamps,
    })

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

    # Plot funding rates for each symbol
    plt.plot(df['Timestamp'], df['Funding Rate'], label=symbol)

plt.title('Funding Rates Over Time - All Symbols')
plt.xlabel('Timestamp')
plt.ylabel('Funding Rate')
plt.legend()
plt.grid(True)
plt.show()


category = 'linear'
limit_newest = 1  # Fetch only the newest funding rate
limit_hist = 100  # Fetch historical funding rates

# Fetch the newest funding rate for all symbols
all_funding_rates = []

for symbol in all_perps:
    # Make the API call to get historical funding rates
    funding_rate_history = session.get_funding_rate_history(category=category, symbol=symbol, limit=limit_newest)

    # Check if the list is not empty before accessing the first element
    if 'list' in funding_rate_history['result'] and funding_rate_history['result']['list']:
        newest_funding_rate = float(funding_rate_history['result']['list'][0]['fundingRate'])
        all_funding_rates.append({'Symbol': symbol, 'Newest Funding Rate': newest_funding_rate})

# Create a DataFrame from the collected data
df_all_funding_rates = pd.DataFrame(all_funding_rates)

# Check if DataFrame is not empty before proceeding
if not df_all_funding_rates.empty:
    # Sort DataFrame by 'Newest Funding Rate' in descending order
    df_all_funding_rates = df_all_funding_rates.sort_values(by='Newest Funding Rate', ascending=False).head(10)

    # Create a single chart with 10 subplots
    fig, axs = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle('Top 10 Symbols - Historical Funding Rates', fontsize=16)

    for i, symbol in enumerate(df_all_funding_rates['Symbol']):
        # Calculate subplot position
        row = i // 2
        col = i % 2

        # Get the current subplot axis
        ax = axs[row, col]

        # Make the API call to get historical funding rates for the selected symbol
        funding_rate_history = session.get_funding_rate_history(category=category, symbol=symbol, limit=limit_hist)

        # Check if the list is not empty before accessing the first element
        if 'list' in funding_rate_history['result'] and funding_rate_history['result']['list']:
            # Extract historical funding rates and timestamps
            funding_rates = [float(entry['fundingRate']) for entry in funding_rate_history['result']['list']]
            timestamps = [int(entry['fundingRateTimestamp']) for entry in funding_rate_history['result']['list']]

            # Convert timestamps to readable format (assuming timestamps are in milliseconds)
            readable_timestamps = [datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S') for timestamp in timestamps]

            # Plot line chart for the selected symbol
            ax.plot(readable_timestamps, funding_rates, label=symbol)
            ax.set_title(symbol)
            ax.set_ylabel('Funding Rate')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)

            # Set x-axis labels for the bottom two charts
            if row == 4:
                ax.set_xlabel('Timestamp')
                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # Set maximum number of date labels
            else:
                ax.set_xticklabels([])  # Hide x-axis labels for other charts

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print('DataFrame is empty. No data to plot.')