import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import nested_lookup as nl
from datetime import datetime
from pybit.unified_trading import HTTP

# fetch biggest funding rates from bybit

# bybit
api_key_by = 'kezeQXNipK0hy06iAd'
api_secret_by = 'DD8F7b8UykIfnDYJ619UMBZaailhtQlWECw2'

session = HTTP(
    testnet=False,
    api_key=api_key_by,
    api_secret=api_secret_by,
)

params = {
    'category': 'linear',
}
# Make the API request
response = session.get_instruments_info(**params)
all_perps = [instrument['symbol'] for instrument in response['result']['list']]

category = 'linear'
limit_newest = 1  # Fetch only the newest funding rate
limit_hist = 300  # Fetch historical funding rates

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

            # reverse timestamps and funding rates
            funding_rates.reverse()
            readable_timestamps.reverse()

            # Plot line chart for the selected symbol
            ax.plot(readable_timestamps, funding_rates, label=symbol)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_title(symbol)
            ax.set_ylabel('Funding Rate')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)

            # Set x-axis labels for the bottom two charts
            if row == 4:
                ax.set_xlabel('Timestamp')
                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
                ax.xaxis.set_major_locator(plt.MaxNLocator(16))  # Display 12 x-axis labels
            else:
                ax.set_xticklabels([])  # Hide x-axis labels for other charts

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print('DataFrame is empty. No data to plot.')