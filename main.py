import os
import sys
import pandas as pd
import numpy as np
from src.load_data import load_data
from src.pairs_trading import find_cointegrated_pairs, calc_dynamic_hedge_ratio_ols, calc_bollinger_ols, dynamic_trading_strategy_pairs_backtest

# get system path


if __name__ == "__main__":
    # get working directory
    cwd = os.getcwd()
    # get data file path
    path = os.path.join(cwd, "src/data/data pairs.csv")
    data = load_data(path)

    pairs = find_cointegrated_pairs(data)

    # choose pair wits smallest p-value in pairs
    tickers_pairs = pairs.iloc[0,0:2]

    # new data frame with only the two tickers
    data_pairs = data[tickers_pairs]

    # trading strategy using dynamic hedge ratio from OLS and upper and lower bound threshold from z-spread
    # sell spread if z-score > threshold
    # buy spread if z-score < threshold

    # params
    window = 20
    threshold = 2

    data_pairs = data_pairs.dropna()

    # calc dynamic hedge ratio
    hedge_ratio, spread_ols = calc_dynamic_hedge_ratio_ols(data_pairs, window=window)
    pd.DataFrame(hedge_ratio).plot()

    # calc z-spread
    spread, z_spread, spread_mean, upper_band, lower_band, hedge_ratio = calc_bollinger_ols(data_pairs, window = window, std_dev = threshold)
    z_spread.plot()

    # backtest strategy
    dynamic_trading_strategy_pairs_backtest(data_pairs, window=window, std_dev= threshold)
