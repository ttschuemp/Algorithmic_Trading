import os
import sys
import pandas
import numpy as np
from src.load_data import load_data
from src.pairs_trading import find_cointegrated_pairs

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


    # first step: find cointegrated pairs and fetch data

    # # test sample using index isin from txt file
    # with open("index_isin.txt") as f:
    #     index_isin = [line.rstrip('\n') for line in f]

    # isin_prefix = '/isin/'

    # # bloomberg data
    # smi_data = blp.bdh([isin_prefix + add for add in index_isin], flds=['PX_CLOSE_1D'], start_date='2020-12-29', Per='D')
    # smi_data.columns = smi_data.columns.get_level_values(0)

    # # find coint pairs for all index member
    # pairs = find_cointegrated_pairs(smi_data)

    # # take pairs with smallest p-value
    # tickers_pairs = pairs.iloc[0,0:2]

    # #data_pairs = blp.bdh(tickers=['EWC US Equity', 'EWA US Equity'], flds=['PX_CLOSE_1D'], start_date='2000-12-29', Per='D')
    # data_pairs.columns = data_pairs.columns.get_level_values(1)
    # data_pairs.columns = ['series1', 'series2']


    # # trading strategy using dynamic hedge ratio from OLS and upper and lower bound threshold from z-spread
    # # sell spread if z-score > threshold
    # # buy spread if z-score < threshold


    # # params
    # window = 20
    # threshold = 2

    # data_pairs = data_pairs.dropna()

    # # calc dynamic hedge ratio
    # hedge_ratio, spread_ols = calc_dynamic_hedge_ratio_ols(data_pairs, window=window)
    # pd.DataFrame(hedge_ratio).plot()

    # # calc z-spread
    # spread, z_spread, spread_mean, upper_band, lower_band, hedge_ratio = calc_bollinger_ols(data_pairs, window = window, std_dev = threshold)
    # z_spread.plot()

    # # backtest strategy
    # dynamic_trading_strategy_pairs_backtest(data_pairs, window=window, std_dev= threshold)
