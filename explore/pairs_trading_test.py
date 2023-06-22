# pairs trading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import blpapi
import scipy
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import requests
from pykalman import KalmanFilter
from scipy import stats
from xbbg import blp
from binance import Client, AsyncClient, ThreadedWebsocketManager, ThreadedDepthCacheManager
from binance.client import Client
from datetime import datetime, timedelta
import time

# helper functions

def fetch_data(ticker, interval, lookback):
    # fetch binance data
    hist_df = pd.DataFrame(client.get_historical_klines(ticker, interval, lookback + 'hours ago UTC'))
    hist_df = hist_df.iloc[:,:6]
    hist_df.columns = ['Time','Open', 'High', 'Low', 'Close', 'Volume']
    hist_df = hist_df.set_index('Time')
    hist_df.index = pd.to_datetime(hist_df.index, unit='ms')
    hist_df = hist_df.astype(float)
    return hist_df


def fetch_crypto_data(top_n, days):
        # Get the top N cryptocurrencies by market cap from CoinGecko API
        response = requests.get(f"https://api.coingecko.com/api/v3/coins/"
                                f"markets?vs_currency=usd&order=market_cap_desc&per_page={top_n}&"
                                f"page=1&sparkline=false&price_change_percentage=24h",
                                verify="C:\DevLab\Zscaler Zertifikat.cer")
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
                if len(klines) >= (days-1)*24:
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


def find_cointegrated_pairs(data):
    ''' find from a list cointegrated pairs'''
    n = data.shape[1]
    keys = data.keys()
    pvalue_matrix = np.ones((n, n))
    pairs = []

    # Loop through each combination of assets
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]

            # Test for cointegration
            result = ts.coint(S1, S2)
            pvalue = result[1]

            # Store p-value in matrix
            pvalue_matrix[i, j] = pvalue

            # Add cointegrated pair to list (if p-value is less than 0.05)
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j], pvalue))

    # Sort cointegrated pairs by p-value in ascending order
    pairs.sort(key=lambda x: x[2])

    return pd.DataFrame(pairs)


def calc_sharpe_ratios(cryptos_coint):
    """backtest all cointegrated pairs and calculate sharpe ratio"""
    pairs = cryptos_coint.iloc[:,0:2]
    results = []

    for _, pair in pairs.iterrows():
        pair_data = crypto_data[list(pair)].dropna().astype(float)
        pair_data.columns = ['series1', 'series2']
        spread, hedge_ratio = calc_spread_ols(pair_data)
        window = int(np.round(half_life(spread)))
        sharpe_ratio = dynamic_trading_strategy_pairs_backtest(pair_data, window=window, std_dev=1)
        results.append({'Pair': tuple(pair), 'Sharpe Ratio': sharpe_ratio})

    sharpe_ratios_df = pd.DataFrame(results)
    return sharpe_ratios_df.sort_values(['Sharpe Ratio'], ascending=[False])



def hurst(df_series):
    """Returns the Hurst exponent of the time series vector ts"""
    # df_series = df_series if not isinstance(df_series, pd.Series) else df_series.to_list()
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = tau = [np.sqrt((df_series - df_series.shift(-lag)).std()) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2


def half_life(df_series):
    """calc half life of time series"""
    df_series = df_series.dropna()
    z_lag = np.roll(df_series, 1)
    z_lag[0] = 0
    z_ret = df_series - z_lag
    z_ret[0] = 0

    # adds intercept terms to X variable for regression
    z_lag2 = sm.add_constant(z_lag)

    model = sm.OLS(z_ret, z_lag2)
    res = model.fit()

    halflife = -np.log(2) / res.params[1]
    return halflife


def test_cointegration(data):
    """calc cointegration of time series"""

    test_stat = coint_johansen(data.dropna(), 0, 1).trace_stat
    # crit values 90%,  95%, 99%
    crit_values = coint_johansen(data.dropna(), 0, 1).trace_stat_crit_vals
    # eigen statistics tells us how strongly cointegrated the series are
    print("Eigenvalues:", coint_johansen(data, 0, 1).max_eig_stat)
    print(print("Critical Values Eigenvalues: ", coint_johansen(data, 0, 1).max_eig_stat_crit_vals))

    # calc the spread from the eigenvectors
    hedge_ratio = coint_johansen(data.dropna(), 0, 1).evec[0, 0] / coint_johansen(data.dropna(), 0, 1).evec[1, 0]
    spread = data.iloc[:, 0] - (hedge_ratio * data.iloc[:, 1])
    adf_result = adfuller(spread.dropna())[1]

    return test_stat, crit_values, spread, adf_result


def pairs_trading_ols(data, std_dev):
    """
    Pairs trading strategy using spread z-score and fix threshold
    # big lock-ahead bias because we calculate the spread over the whole timeframe
    """
    # calc hedge ratio
    x = sm.add_constant(data.iloc[:,1])
    y = data.iloc[:,0]
    model = sm.OLS(y, x).fit()
    hedge_ratio = model.params[1]

    spread = data.iloc[:,0] - hedge_ratio * data.iloc[:,1]

    spread_z = (spread - spread.mean()) / np.std(spread)
    spread_mean = spread_z.mean()
    spread_ub =  1*std_dev
    spread_lb = -1*std_dev
    spread_ub_sl = 1*(std_dev + 1)
    spread_lb_sl =  -1*(std_dev + 1)
    return spread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio, spread_ub_sl, spread_lb_sl



def trading_strategy_pairs_backtest(data, std_dev):

    sspread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio, spread_ub_sl, spread_lb_sl = pairs_trading_ols(data, std_dev)
    data_strategy = data.copy()
    data_strategy['zspread'] = spread_z
    data_strategy['position_long_series1'] = 0
    data_strategy['position_long_series2'] = 0
    data_strategy['position_short_series1'] = 0
    data_strategy['position_short_series2'] = 0

    data_strategy.loc[data_strategy.zspread>=1*std_dev, ('position_short_series1', 'position_short_series2')] = [-1, 1] # Short spread
    data_strategy.loc[data_strategy.zspread<=-1*std_dev, ('position_long_series1', 'position_long_series2')] = [1, -1] # Buy spread
    data_strategy.loc[data_strategy.zspread<=0, ('position_short_series1', 'position_short_series2')] = 0 # Exit short spread
    data_strategy.loc[data_strategy.zspread>=spread_ub_sl, ('position_short_series1', 'position_short_series2')] = 0 # Exit short spread stop loss
    data_strategy.loc[data_strategy.zspread>=0, ('position_long_series1', 'position_long_series2')] = 0 # Exit long spread
    data_strategy.loc[data_strategy.zspread<=spread_lb_sl, ('position_short_series1', 'position_short_series2')] = 0 # Exit long spread stop loss


    data_strategy.fillna(method='ffill', inplace=True) # ensure existing positions are carried forward unless there is an exit signal

    positions_Long = data_strategy.loc[:, ('position_long_series1', 'position_long_series2')]
    positions_Short = data_strategy.loc[:, ('position_short_series1', 'position_short_series2')]
    positions = np.array(positions_Long) + np.array(positions_Short)
    positions = pd.DataFrame(positions)

    # calc returns
    dailyret = data_strategy.loc[:, ('series1', 'series2')].pct_change()
    # calculate pnl
    pnl = (np.array(positions.shift())*np.array(dailyret)).sum(axis=1)
    pnl = pnl[~np.isnan(pnl)]

    # calc sharpe ratio of strategy
    sharpe_ratio = np.sqrt(252) * np.mean(pnl) / np.std(pnl)

    # plot equity curve
    plt.plot(np.cumsum(pnl))
    print(sharpe_ratio)


def calc_spread_ols(data):
    """
    Pairs trading strategy using spread z-score and fix threshold
    # big lock-ahead bias because we calculate the spread over the whole timeframe
    """
    # calc hedge ratio
    x = sm.add_constant(data.iloc[:,1])
    y = data.iloc[:,0]
    model = sm.OLS(y, x).fit()
    hedge_ratio = model.params[1]

    spread = data.iloc[:,0] - hedge_ratio * data.iloc[:,1]
    return spread, hedge_ratio


def calc_dynamic_hedge_ratio_ols(data, window):
    """
    Calculates rolling hedge ratio using OLS
    """
    hedge_ratio = []
    for i in range(window, len(data)):
        # Estimate hedge ratio using OLS
        y = data.iloc[i-window:i,0]
        x = data.iloc[i-window:i,1]
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        hedge_ratio.append(model.params[1])

    spread_ols = data.iloc[window::, 0] - data.iloc[window::, 1] * hedge_ratio

    return hedge_ratio, spread_ols


def calc_bollinger_ols(data, window, std_dev):
    """
   Calculates rolling spread and bollinger bands
    """
    hedge_ratio, spread = calc_dynamic_hedge_ratio_ols(data, window)
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    z_spread = (spread - spread_mean) / spread_std
    upper_band = 1*std_dev
    lower_band = -1*std_dev
    upper_band_sl = 3
    lower_band_sl = -3

    return spread, z_spread, spread_mean, upper_band, lower_band, upper_band_sl, lower_band_sl,  hedge_ratio


def dynamic_trading_strategy_pairs_backtest(data, window, std_dev):

    spread, z_spread, spread_mean, upper_band, lower_band, upper_band_sl, lower_band_sl,  hedge_ratio = calc_bollinger_ols(data, window, std_dev)
    data_strategy = data.copy()
    data_strategy = data_strategy.iloc[window:,:]
    data_strategy['zspread'] = z_spread
    data_strategy['position_long_series1'] = 0
    data_strategy['position_long_series2'] = 0
    data_strategy['position_short_series1'] = 0
    data_strategy['position_short_series2'] = 0

    data_strategy.loc[data_strategy.zspread>=upper_band, ('position_short_series1', 'position_short_series2')] = [-1, 1] # Short spread
    data_strategy.loc[data_strategy.zspread<=lower_band, ('position_long_series1', 'position_long_series2')] = [1, -1] # Buy spread
    data_strategy.loc[data_strategy.zspread<=0, ('position_short_series1', 'position_short_series2')] = 0 # Exit short spread
    data_strategy.loc[data_strategy.zspread>=upper_band_sl, ('position_short_series1', 'position_short_series2')] = 0 # Exit short spread stop loss
    data_strategy.loc[data_strategy.zspread>=0, ('position_long_series1', 'position_long_series2')] = 0 # Exit long spread
    data_strategy.loc[data_strategy.zspread<=lower_band_sl, ('position_short_series1', 'position_short_series2')] = 0 # Exit long spread stop loss

    data_strategy.fillna(method='ffill', inplace=True) # ensure existing positions are carried forward unless there is an exit signal

    positions_Long = data_strategy.loc[:, ('position_long_series1', 'position_long_series2')]
    positions_Short = data_strategy.loc[:, ('position_short_series1', 'position_short_series2')]
    positions = np.array(positions_Long) + np.array(positions_Short)
    positions = pd.DataFrame(positions)

    # calc returns
    dailyret = data_strategy.loc[:, ('series1', 'series2')].pct_change()
    # calculate pnl
    pnl = (np.array(positions.shift())*np.array(dailyret)).sum(axis=1)
    pnl = pnl[~np.isnan(pnl)]

    # calc sharpe ratio of strategy
    sharpe_ratio = np.sqrt(252) * np.mean(pnl) / np.std(pnl)

    # plot equity curve
    plt.plot(np.cumsum(pnl))
    return sharpe_ratio



#%%
# fetch crypto data
api_key = 'MKBNxBnIVjKtxdYGMo130QzS9bdCoZY7YjnnQPjPfRMcBD7MwZQF27ab0lT7FJqL'
api_secret = 'V14OTjBICA4eV0ZgND8LKuuZxXsVZKk7eOccxMuGIOIJcn3ksGih8dOSx3bG0G5i'

client = Client(api_key,api_secret, {"verify": "C:\DevLab\Zscaler Zertifikat.cer", "timeout": 20})

#hist_df = fetch_data("BTCBUSD", '1h', '500 ')
#prices = client.get_all_tickers()

crypto_data = fetch_crypto_data(top_n = 10, days = 90)

cryptos_coint = find_cointegrated_pairs(crypto_data)

backtest_sharpe_ratios = calc_sharpe_ratios(cryptos_coint)

# find coint pairs from all index member

# take pairs with smallest p-value
tickers_pairs = crypto_data[list(backtest_sharpe_ratios.iloc[0,0])]

tickers_pairs = tickers_pairs.dropna()

data_pairs = tickers_pairs.astype(float)

data_pairs.columns = ['series1', 'series2']

#%%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(data_pairs['series1'])
ax2.plot(data_pairs['series2'])

data_pairs = data_pairs.dropna()

# test stationary of series
# H0: non stationary
adfuller(data_pairs['series1'])
adfuller(data_pairs['series2'])

# test cointegration
# H0: no cointegration
# can reject
ts.coint(data_pairs['series1'], data_pairs['series2'])

# cointegration
spread, hedge_ratio = calc_spread_ols(data_pairs)
spread.plot()

# calc half_live
window = int(np.round(half_life(spread), 0 ))

# calc hurst
hurst(spread)

# calc static hedge ratio
spread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio, spread_ub_sl, spread_lb_sl = pairs_trading_ols(data_pairs, std_dev = 1)


plt.figure(figsize=(30,15))
plt.plot(spread_z)
plt.axhline(y=1)
plt.axhline(y=-1)
plt.axhline(y=2)
plt.axhline(y=-2)
plt.axhline(y=3)
plt.axhline(y=-3)
plt.xticks(fontsize=30, color='black')
plt.yticks(fontsize=30, color='black')

# hudge look-ahead bias
trading_strategy_pairs_backtest(data_pairs, std_dev = 1)

# we need forward walking hedge ratio

#%%

# maybe use gridsearch or cross validation for trading time frame or window
# calc forward walking hedge ratio

hedge_ratio, spread_ols = calc_dynamic_hedge_ratio_ols(data_pairs, window = window)
spread_ols.plot()
pd.DataFrame(hedge_ratio).plot()

# stationary, halflife, hurst
adfuller(spread_ols)

# in hours
window_ols = int(np.round(half_life(spread_ols), 0 ))
hurst(spread_ols)

# calc z spread
spread, z_spread, spread_mean, upper_band, lower_band, upper_band_sl, lower_band_sl,  hedge_ratio = calc_bollinger_ols(data_pairs, window = window_ols, std_dev = 1)

plt.figure(figsize=(30,15))
plt.plot(z_spread)
plt.axhline(y=0, color='red')
plt.axhline(y=upper_band, color='red')
plt.axhline(y=lower_band, color='red')
plt.axhline(y=upper_band_sl, color='red', linestyle='dashed', marker='o')
plt.axhline(y=lower_band_sl, color='red', linestyle='dashed', marker='o')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# backtesting
sharpe_ratio = dynamic_trading_strategy_pairs_backtest(data_pairs, window = window_ols, std_dev = 1)


#%%
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def calc_hedge_ratio_johansen(data, window):
    """
    Calculates rolling hedge ratio using Johansen test
    """
    y = data_pairs.iloc[:,0]
    x = data_pairs.iloc[:,1]
    hedge_ratio = []
    for i in range(window, len(y)):
        # perform Johansen test on window of data
        jres = coint_johansen(np.column_stack((y[i-window:i], x[i-window:i])), det_order=0, k_ar_diff=1)
        # extract first eigenvector as hedge ratio
        hr = jres.evec[:, 0][1] / jres.evec[:, 0][0]
        hedge_ratio.append(hr)

    return pd.DataFrame(hedge_ratio)


test_ratio = calc_hedge_ratio(data_pairs, window = window)


#%%
#stepwise updating kalman filter

def kalman_hedge_ratio(data_pairs):
    state_cov_multiplier = np.power(0.01, 2)       # 0.1: spread_std=2.2, cov=16  ==> 0.01: 0.22, 0.16
    observation_cov = 0.001
    means_trace = []
    covs_trace = []
    step = 0

    x = data_pairs['series2'][step]
    y = data_pairs['series1'][step]

observation_matrix_stepwise = np.array([[x, 1]])
observation_stepwise = y
kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                  initial_state_mean=np.ones(2),                      # initial value
                  initial_state_covariance=np.ones((2, 2)),           # initial value
                  transition_matrices=np.eye(2),                      # constant
                  observation_matrices=observation_matrix_stepwise,   # depend on x
                  observation_covariance=observation_cov,                           # constant
                  transition_covariance= np.eye(2)*state_cov_multiplier)                   # constant

state_means_stepwise, state_covs_stepwise = kf.filter(observation_stepwise)             # depend on y

means_trace.append(state_means_stepwise[0])
covs_trace.append(state_covs_stepwise[0])

for step in range(1, data_pairs.shape[0]):
    # print(step)
    x = data_pairs['series2'][step]
    y = data_pairs['series1'][step]
    observation_matrix_stepwise = np.array([[x, 1]])
    observation_stepwise = y

    state_means_stepwise, state_covs_stepwise = kf.filter_update(
        means_trace[-1], covs_trace[-1],
        observation=observation_stepwise,
        observation_matrix=observation_matrix_stepwise)


    means_trace.append(state_means_stepwise.data)
    covs_trace.append(state_covs_stepwise)


means_trace = pd.DataFrame(means_trace, columns = ['Slope', 'Intercept'], index=data_pairs.index)
means_trace.plot(subplots=True)
plt.show()
return means_trace['Slope']


hedge_ratio_kalman = kalman_hedge_ratio(data_pairs)
spread_kalman = data_pairs.iloc[:,0] - hedge_ratio_kalman * data_pairs.iloc[:,1]
hurst(spread_kalman)

#%%

api_key_testnet = '6FVEREC1YOenBUKfisdPXaJHNn8kM3lzxWCgFRDUvyY9fKM2H17pZz6wNg2Spf71'

api_secret_testnet = '0wlHotYWjFIvpZc69FnESbfRhaDUMtcidNJX72obsocGRlH9Feg90rVkT7YCKUqg'


client = Client(api_key_testnet,api_secret_testnet, {"verify": "C:\DevLab\Zscaler Zertifikat.cer", "timeout": 20}, testnet=True)
client.API_URL = 'https://testnet.binance.vision/api'


symbol = 'BTCUSDT'
quantity = 0.01  # The amount of BTC you want to buy
order = client.create_order(symbol=symbol,
                            side = 'BUY',
                            type = 'MARKET',
                            quantity=quantity)
print(order)
# Set the order parameters


#%%
