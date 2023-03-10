
# pairs trading cointegration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import requests
from pykalman import KalmanFilter
from scipy import stats
from binance import Client, AsyncClient, ThreadedWebsocketManager, ThreadedDepthCacheManager
from binance.client import Client
from datetime import datetime, timedelta
import time

# functions for computing trading indicators

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

