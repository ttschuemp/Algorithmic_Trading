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
from pykalman import KalmanFilter
from scipy import stats
from xbbg import blp

#%%

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



def calc_dynamic_hedge_ratio(data, window):
    """
    Calculates rolling hedge ratio using johansen
    """
    hedge_ratio_assets = np.full(data.shape, np.NaN)
    hedge_ratio = np.full(data.shape, np.NaN)

    for i in range(window, len(data)):
        hedge_ratio_assets[i, :] = coint_johansen(data[i - window:i], 0, 1).evec[:, 0].reshape((1, 2))

    hedge_ratio = hedge_ratio_assets[:, 0] / hedge_ratio_assets[:, 1]

    return hedge_ratio_assets[window::], hedge_ratio[window::]


def calc_hedge_ratio_spread_ols(data, window):
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



def calc_spread_bollinger_ols(data, window, std_dev):
    """
    Calculates rolling spread and bollinger bands
    """
    hedge_ratio, spread = calc_hedge_ratio_spread_ols(data, window)
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    upper_band = spread_mean + std_dev * spread_std
    lower_band = spread_mean - std_dev * spread_std

    return spread, spread_mean, upper_band, lower_band, hedge_ratio


def calc_spread_bollinger(data, window, std_dev):
    """
    Calculates rolling spread and bollinger bands
    """
    hedge_ratio_assets, hedge_ratio = calc_dynamic_hedge_ratio(data, window)
    spread = data.iloc[window::, 0] - data.iloc[window::, 1] * hedge_ratio
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    upper_band = spread_mean + std_dev * spread_std
    z_spread = (spread - spread_mean) / spread_std
    lower_band = spread_mean - std_dev * spread_std

    return spread, z_spread, upper_band, lower_band, hedge_ratio, hedge_ratio_assets


def pairs_trading_bollinger_bands_ols(data, window, std_dev):
    """
    Pairs trading strategy using bollinger bands
    """
    spread, z_spread, upper_band, lower_band, hedge_ratio = calc_spread_bollinger_ols(data, window, std_dev)
    signals = np.zeros(len(z_spread))
    signals[z_spread > upper_band] = -1
    signals[z_spread < lower_band] = 1
    signals[(z_spread > lower_band) & (z_spread < upper_band)] = 0
    signals = pd.Series(signals, name="signals")
    hedge_ratio = pd.Series(hedge_ratio, name="hedge ratio")
    signals.index = z_spread.index
    hedge_ratio.index = z_spread.index
    result = pd.concat([spread, z_spread, upper_band, lower_band, signals, hedge_ratio], axis=1)
    result.columns = ['spread', 'z_spread', 'upper_band', 'lower_band', 'signals', 'hedge_ratio']
    return result


def pairs_trading_bollinger_bands(data, window, std_dev):
    """
    Pairs trading strategy using bollinger bands
    """
    spread, z_spread, upper_band, lower_band, hedge_ratio, hedge_ratio_assets = calc_spread_bollinger(data, window, std_dev)
    signals = np.zeros(len(z_spread))
    # sell the spread - 1
    signals[z_spread > upper_band] = -1
    # buy the spread 1
    signals[z_spread < lower_band] = 1
    signals[(z_spread > lower_band) & (z_spread < upper_band)] = 0
    signals = pd.Series(signals, name="signals")
    hedge_ratio = pd.Series(hedge_ratio, name="hedge ratio")
    signals.index = z_spread.index
    hedge_ratio.index = z_spread.index
    result = pd.concat([spread, z_spread, upper_band, lower_band, signals, hedge_ratio], axis=1)
    result.columns = ['spread', 'z_spread', 'upper_band', 'lower_band', 'signals', 'hedge_ratio']
    return result


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
    return spread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio


def trading_strategy_pairs_backtest(data, std_dev):

    spread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio = pairs_trading_ols(data, std_dev)
    data_strategy = data.copy()
    data_strategy['zspread'] = spread_z
    data_strategy['position_long_series1'] = 0
    data_strategy['position_long_series2'] = 0
    data_strategy['position_short_series1'] = 0
    data_strategy['position_short_series2'] = 0

    data_strategy.loc[data_strategy.zspread>=1*std_dev, ('position_short_series1', 'position_short_series2')] = [-1, 1] # Short spread
    data_strategy.loc[data_strategy.zspread<=-1*std_dev, ('position_long_series1', 'position_long_series2')] = [1, -1] # Buy spread
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
# first step: find cointegrated pairs
# problem false positive with multiple testing
# problem is correlation/cointegration values != causation
# cointegration without causation may break someday
# Cointegrated pairs trading should be applied when there
# really is a fundamental linkage between the underlying assets
# so we have to look for pairs within same sector or with causation

# in crypto maybe us fast intraday intraday relationships or 1h,4h time ticks
# try more than two securities
# try running a monte carlo on the results to see all possible scenarios

# test sample using index isin from txt file
with open("index_isin.txt") as f:
    index_isin = [line.rstrip('\n') for line in f]

isin_prefix = '/isin/'

# bloomberg data
smi_data = blp.bdh([isin_prefix + add for add in index_isin], flds=['PX_CLOSE_1D'], start_date='2000-12-29', Per='D')
smi_data.columns = smi_data.columns.get_level_values(0)
smi_data  = smi_data.dropna()
# drop nan
#smi_data = smi_data.dropna(axis=1)

# find coint pairs from all index member
pairs = find_cointegrated_pairs(smi_data)

# take pairs with smallest p-value
tickers_pairs = pairs.iloc[0,0:2]

# cryptos
crypto = ['XBTUSD BGNL Curncy', 'XETUSD Curncy', 'XRPUSD BGNL Curncy', 'XSOUSD Curncy', 'XBI DAR Curncy',
          'XAD BGNL Curncy', 'XMA Curncy']

crypto_data = blp.bdh(tickers=crypto, flds=['PX_LAST'], start_date='2000-12-29', Per='D')

pairs_crypto = find_cointegrated_pairs(crypto_data)

#data_pairs = blp.bdh(tickers=tickers_pairs, flds=['PX_CLOSE_1D'], start_date='2000-12-29', Per='D')
#data_pairs = blp.bdh(tickers=['EWC US Equity', 'EWA US Equity'], flds=['PX_CLOSE_1D'], start_date='2000-12-29', Per='D')
#data_pairs = blp.bdh(tickers=['EZJ LN Equity', 'RYA ID Equity'], flds=['PX_CLOSE_1D'], start_date='2000-12-29', Per='D')
data_pairs = blp.bdh(tickers=['ABX CT Equity', 'AEM CN Equity'], flds=['PX_CLOSE_1D'], start_date='2000-12-29', Per='D')


data_pairs.columns = data_pairs.columns.get_level_values(1)
data_pairs.columns = ['series1', 'series2']


#%%
# strategy using johansen test

data_pairs = data_pairs.dropna()
data_log = np.log(data_pairs)

# plot chart
data_pairs.plot()
data_log.plot()

# test for cointegration
# H0: not cointegrated
# johansen test
# first H0: r = 0 -> no cointegration relationship between the series. we can reject this H0
# second H0: r <= 1 -> one or less cointegration relationship. we can reject this H0 on 90% level
# can reject first H0 and second H0 for 90% level
# if we can reject all H0 -> number of cointegrating relationship being equal of the number of time series
# on 90% level both series are cointegrated

# ADF test H0: non stationary
# spread is not stationary and therefore not mean reverting
test_stat, crit_values, spread, adf_result_pvalue = test_cointegration(data_pairs)
spread.plot()

# calc hurst exponent of series
# 0.484 < 0.5 -> hurst indicates mean reversion of spread but ADF test not
hurst_exp = hurst(spread)

# calc half life
# 1449 def to long
half_life(spread)

# trading strategy using hedge ratio from OLS and upper and lower bound threshold from z-spread
# sell spread if z-score > threshold
# buy spread if z-score < threshold
spread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio = pairs_trading_ols(data_pairs, std_dev=1)

# backtest strategy
trading_strategy_pairs_backtest(data_pairs, std_dev=1)

# this strategy has a huge look a head bias, because we calculate the optimal hedge ratio based on the entire data set

fig, ax = plt.subplots(figsize=(25, 15))
plt.plot(spread_z, label="spread mean")
plt.axhline(spread_ub, label="Upper Band")
plt.axhline(spread_lb , label="Lower Band")
plt.axhline(spread_mean , label="mean")
# plt.plot(spread_ols, label="spread")
plt.title("Spread")
plt.legend()
plt.show()

# split data in test and training set
train_set, test_set = np.split(data_pairs, [int(0.75 *len(data_pairs))])

trading_strategy_pairs_backtest(train_set, std_dev=1)
trading_strategy_pairs_backtest(test_set, std_dev=1)

# we need a forward walking hedge ratio with a backwards looking window


#%%
# plot buy and sell signals of the spread
plt.figure(figsize=(25, 15))
spread.plot()
buy = spread.copy()
sell = spread.copy()
buy[spread_z>-1] = 0
sell[spread_z<1] = 0
buy.plot(color='g', linestyle='None', marker='^')
sell.plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, spread.min(), spread.max()))
plt.legend(['Buy Signal', 'Sell Signal'])
plt.show()


plt.figure(figsize=(25, 15))
S1 = data_pairs.iloc[:,0]
S2 = data_pairs.iloc[:,1]

S1.plot(color='b')
S2.plot(color='c')
buyR = 0*S1.copy()
sellR = 0*S1.copy()

# When you buy the ratio, you buy stock S1 and sell S2
buyR[buy!=0] = S1[buy!=0]
sellR[buy!=0] = S2[buy!=0]

# When you sell the ratio, you sell stock S1 and buy S2
buyR[sell!=0] = S2[sell!=0]
sellR[sell!=0] = S1[sell!=0]

buyR.plot(color='g', linestyle='None', marker='^')
sellR.plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))

plt.legend(['series 1', 'series 2', 'Buy Signal', 'Sell Signal'])
plt.show()


#%%
# trading strategy bollinger bands using dynamic hedge ratio from johansen test

# calc dynamic hedge ratio
hedge_ratio_assets, hedge_ratio = calc_dynamic_hedge_ratio(data_pairs, window=30)

# Calculates rolling spread and bollinger bands with rolling window and std threshold
spread_dynamic, spread_z, upper_band, lower_band, hedge_ratio_dynamic, hedge_ratio_assets = \
    calc_spread_bollinger(data_pairs, window=30, std_dev=1)

spread_dynamic.plot()
spread_z.plot()


#%%
# log series
# try total least squares regression
# try Kalman Filter

# change johannsen test to OLS because of the large spread spikes
# or change spread calc, don't use e1/e2, becuase if e2 is small, spread becomes huge
# maybe scale the hedge ratio per unit * 100 and then calc the hedge ratio
# maybe give this a try https://robotwealth.com/practical-pairs-trading/

# calc spread whole dataset
spread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio = pairs_trading_ols(data_pairs, std_dev = 1)
spread.plot()

# H0: spread is non stationary
# can't reject
adfuller(spread)

# half live for mean reverting
# 292 days - way to long
half_life(spread)

# hurst
hurst(spread)

# use OLS to calculate dynamic hedge ratio and spread
hedge_ratio_ols, spread_ols = calc_hedge_ratio_spread_ols(data_pairs, window = 20)
spread_ols.plot()
pd.DataFrame(hedge_ratio_ols).plot()

# test stationary of cointegrated serie
# adf test would suggest highly stationary serie
adfuller(spread_ols)

# calc half life
# half life is 22 days
half_life(spread_ols)

# trading strategy using bollinger bands and dynamic ols hedge ratio
# spread = Y - hedge_ratio * X
# normalize the spread to z-spread
# if z-spread > upper band: Sell 1 unit of Y and Buy hedge_ratio * X units
# if z-spread < lower band: Buy 1 unit of Y and Sell hedge_ratio * X units

# calculate bollinger bands
spread_ols_dynamic, spread_z_ols, upper_band_ols, lower_band_ols, hedge_ratio_ols_dynamic \
    = calc_spread_bollinger_ols(data_pairs, window=24, std_dev=2)

spread_z_ols.plot()

results = pairs_trading_bollinger_bands_ols(data_pairs, window = 22, std_dev = 2)

fig, ax = plt.subplots(figsize=(25, 15))
plt.plot(spread_z_ols, label="spread mean")
plt.plot(upper_band_ols, label="Upper Band")
plt.plot(lower_band_ols , label="Lower Band")
# plt.plot(spread_ols, label="spread")
plt.title("Bollinger Bands")
plt.legend()
plt.show()

#%%
# pairs trading with walk forward hedge ratio

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
# start here!

data_pairs = data_pairs.dropna()

# calc spread whole dataset
spread, spread_z, spread_mean, spread_ub, spread_lb, hedge_ratio = pairs_trading_ols(data_pairs, std_dev = 1)
spread.plot()

# H0: spread is non stationary
# can't reject
adfuller(spread)

# half live for mean reverting
# 292 days - way to long
half_life(spread)


# calc dynamic hedge ratio
# problem hedge ratio really noisy -> test kalman filter or EWA to smooth hedge ratio
hedge_ratio, spread_ols = calc_dynamic_hedge_ratio_ols(data_pairs, window=24)
pd.DataFrame(hedge_ratio).plot()

# calc z-spread
spread, z_spread, spread_mean, upper_band, lower_band, hedge_ratio = calc_bollinger_ols(data_pairs, window = 24, std_dev = 1)
z_spread.plot()

# backtest strategy
dynamic_trading_strategy_pairs_backtest(data_pairs, window=24, std_dev=1)


fig, ax = plt.subplots(figsize=(25, 15))
plt.plot(z_spread, label="spread mean")
plt.plot(upper_band, label="Upper Band")
plt.plot(lower_band , label="Lower Band")
# plt.plot(spread_ols, label="spread")
plt.title("Bollinger Bands")
plt.legend()
plt.show()


# use Kalman filter fo calc dynamic hedge ratio
# https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter/

# or try grid search with ols

#%%
def kalman_simple(data):
    """ using simple kalman filter"""
    H = np.eye(2)
    delta = 1e-5
    vt = 0.1
    Wt = delta / (1 - delta) * np.eye(2)
    R = np.ones((2,2))
    theta = np.zeros(2)

    F = np.vstack([data.iloc[:, 0], np.ones(data.iloc[:, 0].shape)]).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=2,
        initial_state_mean=theta,
        initial_state_covariance=R,
        transition_matrices=H,
        observation_matrices=F,
        observation_covariance=vt,
        transition_covariance=Wt,
    )

    # State means are frequently represented by theta
    state_means, state_covs = kf.filter(data.iloc[:, 1].values)
    means_trace = pd.DataFrame(state_means, columns = ['slope', 'intercept'], index=data.index)
    spread_kalman = data['series1'] - (means_trace['slope'] * data['series2'] + means_trace['intercept'])

    return means_trace['slope'], spread_kalman

def stepwise_hedge_ratio_kalman(data):
    """ calc stepwise updating kalman filter for dynamic hedge ratio"""
    state_cov_multiplier = np.power(0.01, 2)
    observation_cov = 0.001
    means_trace = []
    covs_trace = []
    step = 0

    x = data['series2'][step]
    y = data['series1'][step]

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

    for step in range(1, data.shape[0]):
        # print(step)
        x = data['series2'][step]
        y = data['series1'][step]
        observation_matrix_stepwise = np.array([[x, 1]])
        observation_stepwise = y

        state_means_stepwise, state_covs_stepwise = kf.filter_update(
            means_trace[-1], covs_trace[-1],
            observation=observation_stepwise,
            observation_matrix=observation_matrix_stepwise)


        means_trace.append(state_means_stepwise.data)
        covs_trace.append(state_covs_stepwise)


    means_trace = pd.DataFrame(means_trace, columns = ['slope', 'intercept'], index=data.index)
    spread_kalman = data['series1'] - (means_trace['slope'] * data['series2'] + means_trace['intercept'])
    #spread_kalman_2 = data_pairs['series1'] - means_trace['slope'] * data_pairs['series2'] - means_trace['intercept']
    #spread_kalman_3 = np.log(data_pairs['series1']) - means_trace['slope'] * np.log(data_pairs['series2']) - means_trace['intercept']

    return means_trace['slope'], spread_kalman


def calc_threshold_kalman(data, std_dev, simple):
    """
    Calculates threshold
    """
    if simple == True:
        hedge_ratio_kalman, spread_kalman = kalman_simple(data)

    else:
        hedge_ratio_kalman, spread_kalman = stepwise_hedge_ratio_kalman(data)

    spread_mean = spread_kalman.mean()
    upper_band = spread_mean + std_dev * spread_kalman.std()
    lower_band = spread_mean - std_dev * spread_kalman.std()

    return hedge_ratio_kalman, spread_kalman, spread_mean, upper_band, lower_band


def kalman_trading_strategy_pairs_backtest(data, std_dev, simple):

    hedge_ratio_kalman, spread_kalman, spread_mean, spread_ub, spread_lb = calc_threshold_kalman(data, std_dev, simple)
    data_strategy = data.copy()
    data_strategy['spread_kalman'] = spread_kalman
    data_strategy['position_long_series1'] = 0
    data_strategy['position_long_series2'] = 0
    data_strategy['position_short_series1'] = 0
    data_strategy['position_short_series2'] = 0

    data_strategy.loc[data_strategy.spread_kalman>=spread_ub, ('position_short_series1', 'position_short_series2')] = [-1, 1] # Short spread
    data_strategy.loc[data_strategy.spread_kalman<=spread_lb, ('position_long_series1', 'position_long_series2')] = [1, -1] # Buy spread
    data_strategy.loc[data_strategy.spread_kalman<=0, ('position_short_series1', 'position_short_series2')] = 0 # Exit short spread
    data_strategy.loc[data_strategy.spread_kalman>=0, ('position_long_series1', 'position_long_series2')] = 0 # Exit long spread
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
data_log = np.log(data_pairs)
# example using simple kalma filter
hedge_ratio_kalman, spread_kalman = kalman_simple(data_pairs)
hedge_ratio_kalman.plot()
spread_kalman.plot()

# maybe use zscore for trading signal with window from half life from kalman. half life can optained from kalman

hedge_ratio_kalman, spread_kalman, spread_mean, upper_band, lower_band = calc_threshold_kalman(data_pairs, std_dev=1, simple=True)

fig, ax = plt.subplots(figsize=(25, 15))
plt.plot(spread_kalman, label="spread kalman")
plt.axhline(spread_mean, label="mean")
plt.axhline(upper_band , label="Upper Band")
plt.axhline(lower_band , label="Lower Band")
plt.title("kalman spread")
plt.legend()
plt.show()

# test stationary of cointegrated serie
# adf test would suggest not stationary time series
adfuller(spread_kalman)

# calc half life
# half life 631 is too long
half_life(spread_kalman)

# hurst
# 0.479 indicates mean reversion
hurst(spread_kalman)

kalman_trading_strategy_pairs_backtest(data_pairs, std_dev=1, simple=True)



#%%

# using stepwise updating hedge ratio with kalman filter

log_pairs = np.log(data_pairs)

hedge_ratio_kalman, spread_kalman = stepwise_hedge_ratio_kalman(data_pairs)
hedge_ratio_kalman.plot()
spread_kalman.plot()

hedge_ratio_kalman, spread_kalman, spread_mean, upper_band, lower_band = calc_threshold_kalman(data_pairs, std_dev=1, simple=False)

fig, ax = plt.subplots(figsize=(25, 15))
plt.plot(spread_kalman, label="spread kalman")
plt.axhline(spread_mean, label="mean")
plt.axhline(upper_band , label="Upper Band")
plt.axhline(lower_band , label="Lower Band")
plt.title("kalman spread")
plt.legend()
plt.show()

# test stationary of cointegrated serie
# adf test would suggest highly stationary serie
adfuller(spread_kalman)

# calc half life
half_life(spread_kalman)

# hurst
hurst(spread_kalman)


kalman_trading_strategy_pairs_backtest(data_pairs, std_dev=1, simple=False)



# diffrent spread calculation
#e=yt−y^

# log prices vs normal prices?
# how to calc spread
# backtest with out of sample
# use z spread rolling window




#%%
import backtrader as bt

df = blp.bdh(tickers=['EZJ LN Equity'], flds=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'], start_date='2000-12-29', Per='D')
df.columns = df.columns.get_level_values(1)
data = df
data.index = pd.to_datetime(data.index,format="%Y-%m-%d",utc=True)
data = data.dropna()
#%%

# https://stackoverflow.com/questions/63471764/importerror-cannot-import-name-warnings-from-matplotlib-dates/66871735#66871735
# https://blog.quantinsti.com/backtrader/#:~:text=%E2%9D%A4%20by%20GitHub-,How%20to%20backtest%20a%20strategy%20with%20Backtrader%3F,components%20of%20the%20strategy%20class.


class MAstrategy(bt.Strategy):
    # when initializing the instance, create a 100-day MA indicator using the closing price
    def __init__(self):
        self.ma = bt.indicators.SimpleMovingAverage(self.data.close, period=100)
        self.order = None

    def next(self):
        if self.order:
            return
        if not self.position: # check if you already have a position in the market
            if (self.data.close[0] > self.ma[0]) & (self.data.close[-1] < self.ma[-1]):
                self.log('Buy Create, %.2f' % self.data.close[0])
                self.order = self.buy(size=10) # buy when closing price today crosses above MA.
            if (self.data.close[0] < self.ma[0]) & (self.data.close[-1] > self.ma[-1]):
                self.log('Sell Create, %.2f' % self.data.close[0])
                self.order = self.sell(size=10)  # sell when closing price today below MA
        else:
            # This means you are in a position, and hence you need to define exit strategy here.
            if len(self) >= (self.bar_executed + 4):
                self.log('Position Closed, %.2f' % self.data.close[0])
                self.order = self.close()

    # outputting information
    def log(self, txt):
        dt=self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    "Executed BUY (Price: %.2f, Value: %.2f, Commission %.2f)" %
                    (order.executed.price, order.executed.value, order.executed.comm))
            else:
                self.log(
                    "Executed SELL (Price: %.2f, Value: %.2f, Commission %.2f)" %
                    (order.executed.price, order.executed.value, order.executed.comm))
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None


if __name__ == '__main__':
    # Create a cerebro instance, add our strategy, some starting cash at broker and a 0.1% broker commission
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MAstrategy)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data)

    print('<START> Brokerage account: $%.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('<FINISH> Brokerage account: $%.2f' % cerebro.broker.getvalue())
    %matplotlib inline
    # Plot the strategy
    cerebro.plot(style='candlestick',loc='grey', volume = False) #You can leave inside the paranthesis empty





#%%
df = data_pairs

cerebro = bt.Cerebro()
df = bt.feeds.PandasData(dataname=df)
cerebro.adddata(df)

class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    def __init__(self):
        sma1 = bt.ind.SMA(period=50)  # fast moving average
        sma2 = bt.ind.SMA(period=100)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position


cerebro.addstrategy(SmaCross)
cerebro.run()
cerebro.plot()
#%%
