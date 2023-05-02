# pairs trading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import blpapi
import scipy
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
from hurst import compute_Hc, random_walk


# function for pairs trading with walk forward hedge ratio

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

def find_cointegrated_pairs_hurst(data):
    ''' find from a list cointegrated pairs'''
    n = data.shape[1]
    keys = data.keys()
    pvalue_matrix = np.ones((n, n))
    pairs = []

    # Loop through each combination of assets
    for i in range(n):
        for j in range(i+1, n):
            S1 = pd.to_numeric(data[keys[i]])
            S2 = pd.to_numeric(data[keys[j]])

            # Test for cointegration
            result = ts.coint(S1, S2)
            pvalue = result[1]

            # Store p-value in matrix
            pvalue_matrix[i, j] = pvalue

            # Add cointegrated pair to list (if p-value is less than 0.05)
            if pvalue < 0.05:

                # calc half life
                model = sm.OLS(S1, S2)
                results = model.fit()
                hedgeRatio = results.params
                z = S1 - hedgeRatio[0] * S2
                prevz = z.shift()
                dz = z-prevz
                dz = dz[1:,]
                prevz = prevz[1:,]
                model2 = sm.OLS(dz, prevz-np.mean(prevz))
                results2 = model2.fit()
                theta = results2.params
                half_life = -np.log(2)/theta

                lags = range(2, len(z)//2)
                tau = [np.sqrt(np.abs(pd.Series(z) - pd.Series(z).shift(lag)).dropna().var()) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst_exp = poly[0] * 2

                hurst_2, c, data = compute_Hc(z, kind='change', simplified=True)


                # calc hurst exponent
                if hurst_exp < 0.5:
                    pairs.append((keys[i], keys[j], pvalue, half_life.values, hurst_exp, hurst_2))


    # Sort cointegrated pairs by p-value in ascending order
    pairs.sort(key=lambda x: x[2])

    return pd.DataFrame(pairs, columns=['Asset 1', 'Asset 2', 'P-value', 'Half Life', 'Hurst', 'Hurst 2'])

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
    upper_band = spread_mean + std_dev * spread_std
    lower_band = spread_mean - std_dev * spread_std

    return spread, z_spread, spread_mean, upper_band, lower_band, hedge_ratio

def dynamic_trading_strategy_pairs_backtest(data, window, std_dev):

    spread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio = calc_bollinger_ols(data, window, std_dev)
    data_strategy = data.copy()
    data_strategy = data_strategy.iloc[window:,:]
    data_strategy['zspread'] = spread_z
    data_strategy['position_long_series1'] = 0
    data_strategy['position_long_series2'] = 0
    data_strategy['position_short_series1'] = 0
    data_strategy['position_short_series2'] = 0

    data_strategy.loc[data_strategy.zspread>=spread_ub, ('position_short_series1', 'position_short_series2')] = [-1, 1] # Short spread
    data_strategy.loc[data_strategy.zspread<=spread_lb, ('position_long_series1', 'position_long_series2')] = [1, -1] # Buy spread
    data_strategy.loc[data_strategy.zspread<=0, ('position_short_series1', 'position_short_series2')] = 0 # Exit short spread
    data_strategy.loc[data_strategy.zspread>=0, ('position_long_series1', 'position_long_series2')] = 0 # Exit long spread
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


#%%
