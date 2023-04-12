# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

import pandas as pd
from src.pairs_trading_functions import find_cointegrated_pairs
from src.load_data import fetch_crypto_data, fetch_data
from binance import Client
from src.api_key_secret import api_key, api_secret, path_zert

#%%
class PairTradingStrategy(bt.Strategy):


    def __init__(self, window, std_dev, status,
                 portfolio_value, period, printout=True):
        # To control parameter entries
        self.orderid = None
        #self.qty1 = qty1
        #self.qty2 = qty2
        self.window = window
        self.std_dev = std_dev
        #self.upper_limit = upper
        #self.lower_limit = lower
        #self.up_medium = up_medium
        #self.low_medium = low_medium
        self.status = status
        self.portfolio_value = portfolio_value
        self.data0 = self.datas[0]
        self.data1 = self.datas[1]
        self.period = period
        self.printout = printout
        self.spread = None
        self.zscore = None
        self.upper_bound = None
        self.lower_bound = None
        self.hedge_ratio = []
        self.size = size


    def log(self, txt, dt=None):
        if self.printout:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return  # Await further notifications

        if order.status == order.Completed:
            if order.isbuy():
                buytxt = 'BUY COMPLETE, %.2f' % order.executed.price
                self.log(buytxt, order.executed.dt)
            else:
                selltxt = 'SELL COMPLETE, %.2f' % order.executed.price
                self.log(selltxt, order.executed.dt)

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log('%s ,' % order.Status[order.status])
            pass  # Simply log

        # Allow new orders
        self.orderid = None


    def calc_dynamic_hedge_ratio_ols(self, data0, data1):
        hedge_ratio = []
        window = self.window
        for i in range(window, len(data0)):
            # Estimate hedge ratio using OLS
            y = data0.iloc[i-self.window:i,0]
            x = data1.iloc[i-self.window:i,1]
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            hedge_ratio.append(model.params[1])

        spread_ols = data0.iloc[window::, 0] - data1.iloc[window::, 1] * hedge_ratio

        return hedge_ratio, spread_ols


    def calc_bollinger_ols(self, data0, data1):
        window = self.window
        std_dev = self.std_dev
        hedge_ratio, spread = self.calc_dynamic_hedge_ratio_ols(data0, data1)
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std()
        z_spread = (spread - spread_mean) / spread_std
        upper_band = spread_mean + std_dev * spread_std
        lower_band = spread_mean - std_dev * spread_std

        return spread, z_spread, spread_mean, upper_band, lower_band, hedge_ratio


    def next(self):

        if self.orderid:
            return  # if an order is active, no new orders are allowed

        if self.printout:
            print('Self  len:', len(self))
            print('Data0 len:', len(self.data0))
            print('Data1 len:', len(self.data1))
            print('Data0 len == Data1 len:',
                  len(self.data0) == len(self.data1))

            print('Data0 dt:', self.data0.datetime.datetime())
            print('Data1 dt:', self.data1.datetime.datetime())

        print('status is', self.status)
        print('zscore is', self.zscore[0])

        # Step 2: Check conditions for SHORT & place the order
        # Checking the condition for SHORT

        data0 = self.datas[0]
        data1 = self.datas[1]

        hedge_ratio = self.hedge_ratio[-1]
        self.spread = data0.Close[0] - hedge_ratio * data1.Close[0]
        self.zscore = (self.spread - self.upper_bound[-1]) / self.spread_std


        if (self.zscore[0] > self.upper_limit) and (self.status != 1):

            # Calculating the number of shares for each stock
            value = 0.5 * self.portfolio_value  # Divide the cash equally
            x = int(value / (self.data0.close))  # Find the number of shares for Stock1
            y = int(value / (self.data1.close))  # Find the number of shares for Stock2
            print('x + self.qty1 is', x + self.qty1)
            print('y + self.qty2 is', y + self.qty2)

            # Placing the order
            self.log('SELL CREATE %s, price = %.2f, qty = %d' % ("PEP", self.data0.close[0], x + self.qty1))
            self.sell(data=self.data0, size=(x + self.qty1))  # Place an order for buying y + qty2 shares
            self.log('BUY CREATE %s, price = %.2f, qty = %d' % ("KO", self.data1.close[0], y + self.qty2))
            self.buy(data=self.data1, size=(y + self.qty2))  # Place an order for selling x + qty1 shares

            # Updating the counters with new value
            self.qty1 = x  # The new open position quantity for Stock1 is x shares
            self.qty2 = y  # The new open position quantity for Stock2 is y shares

            self.status = 1  # The current status is "short the spread"

            # Step 3: Check conditions for LONG & place the order
            # Checking the condition for LONG
        elif (self.zscore[0] < self.lower_limit) and (self.status != 2):

            # Calculating the number of shares for each stock
            value = 0.5 * self.portfolio_value  # Divide the cash equally
            x = int(value / (self.data0.close))  # Find the number of shares for Stock1
            y = int(value / (self.data1.close))  # Find the number of shares for Stock2
            print('x + self.qty1 is', x + self.qty1)
            print('y + self.qty2 is', y + self.qty2)

            # Place the order
            self.log('BUY CREATE %s, price = %.2f, qty = %d' % ("PEP", self.data0.close[0], x + self.qty1))
            self.buy(data=self.data0, size=(x + self.qty1))  # Place an order for buying x + qty1 shares
            self.log('SELL CREATE %s, price = %.2f, qty = %d' % ("KO", self.data1.close[0], y + self.qty2))
            self.sell(data=self.data1, size=(y + self.qty2))  # Place an order for selling y + qty2 shares

            # Updating the counters with new value
            self.qty1 = x  # The new open position quantity for Stock1 is x shares
            self.qty2 = y  # The new open position quantity for Stock2 is y shares
            self.status = 2  # The current status is "long the spread"


            # Step 4: Check conditions for No Trade
            # If the z-score is within the two bounds, close all
        """
        elif (self.zscore[0] < self.up_medium and self.zscore[0] > self.low_medium):
            self.log('CLOSE LONG %s, price = %.2f' % ("PEP", self.data0.close[0]))
            self.close(self.data0)
            self.log('CLOSE LONG %s, price = %.2f' % ("KO", self.data1.close[0]))
            self.close(self.data1)
        """

    def stop(self):
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')

#%%

import backtrader as bt
import backtrader.indicators as btind
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import pandas as pd


class PairsTrading(bt.Strategy):
    params = (
        ("window", 30),
        ("std_dev", 2),
    )

    def __init__(self):
        self.data_a = self.datas[0]
        self.data_b = self.datas[1]


        self.spread = None
        self.z_spread = None
        self.spread_mean = None
        self.upper_band = None
        self.lower_band = None
        self.hedge_ratio = None


    """def log(self, txt, dt=None):
        if self.printout:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return  # Await further notifications

        if order.status == order.Completed:
            if order.isbuy():
                buytxt = 'BUY COMPLETE, %.2f' % order.executed.price
                self.log(buytxt, order.executed.dt)
            else:
                selltxt = 'SELL COMPLETE, %.2f' % order.executed.price
                self.log(selltxt, order.executed.dt)

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log('%s ,' % order.Status[order.status])
            pass  # Simply log

        # Allow new orders
        self.orderid = None
    """

    def calc_dynamic_hedge_ratio_ols(self, data, window):
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

    def calc_bollinger_ols(self, data, window, std_dev):
        """
        Calculates rolling spread and bollinger bands
        """
        hedge_ratio, spread = self.calc_dynamic_hedge_ratio_ols(data, window)
        self.hedge_ratio = hedge_ratio

        self.spread_mean = spread.rolling(window).mean()
        self.spread_std = spread.rolling(window).std()
        self.upper_band = self.spread_mean + std_dev * self.spread_std
        self.lower_band = self.spread_mean - std_dev * self.spread_std
        self.z_spread = (spread - self.spread_mean) / self.spread_std

        return spread, self.z_spread, self.spread_mean, self.upper_band, self.lower_band, hedge_ratio

    def next(self):
        """
        if self.orderid:
            return  # if an order is active, no new orders are allowed

        if self.printout:
            print('Self  len:', len(self))
            print('Data0 len:', len(self.data0))
            print('Data1 len:', len(self.data1))
            print('Data0 len == Data1 len:',
                  len(self.data0) == len(self.data1))

            print('Data0 dt:', self.data0.datetime.datetime())
            print('Data1 dt:', self.data1.datetime.datetime())

        print('status is', self.status)
        print('zscore is', self.z_spread[0])
        """

        # Calculate the dynamic hedge ratio and Bollinger bands
        data = pd.DataFrame([data_df0.Close, data_df1.Close]).T
        spread, z_spread, spread_mean, upper_band, lower_band, hedge_ratio = \
            self.calc_bollinger_ols(data, self.params.window, self.params.std_dev)

        self.spread = spread
        self.z_spread = z_spread
        self.spread_mean = spread_mean
        self.upper_band = upper_band
        self.lower_band = lower_band
        self.hedge_ratio = hedge_ratio
        print(z_spread[-1])

        # Sell spread if zscore is higher than upper bound
        if z_spread[-1] > upper_band[-1]:
            self.sell(data=self.data_a, size=self.hedge_ratio[-1])
            self.buy(data=self.data_b)

        # Buy spread if zscore is lower than lower bound
        elif z_spread[-1] < lower_band[-1]:
            self.buy(data=self.data_a, size=self.hedge_ratio[-1])
            self.sell(data=self.data_b)


#%%
import collections

class PairsTrading(bt.Strategy):
    params = (
        ("window", 30),
        ("std_dev", 1),
        ("size", 1000)
    )

    def __init__(self):
        self.data_a = self.datas[0].close
        self.data_b = self.datas[1].close
        self.hedge_ratio = None
        self.window = self.params.window
        self.zscore = None
        self.spread_history = collections.deque(maxlen=self.params.window)
        self.upper_bound = self.params.std_dev
        self.lower_bound = -self.params.std_dev
        self.size = self.params.size

        self.ols_slope = btind.OLS_Slope_InterceptN(self.data_a, self.data_b, period=self.params.window)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print("{} {}".format(dt.isoformat(), txt))

    def calc_hedge_ratio(self):

        hedge_ratio = self.ols_slope.slope[0]
        spread = self.data_a[0] - (hedge_ratio * self.data_b[0])
        self.spread_history.append(spread)
        spread_mean = pd.Series(self.spread_history).rolling(self.params.window).mean().iloc[-1]
        spread_std_dev = pd.Series(self.spread_history).rolling(self.params.window).std().iloc[-1]
        self.zscore = (spread - spread_mean) / spread_std_dev
        self.hedge_ratio = hedge_ratio
        #print((spread - spread_mean) / spread_std_dev)


    def next(self):
        self.calc_hedge_ratio()

        # Check if there is already an open trade
        if self.position.size == 0:
            if self.zscore < self.lower_bound:
                # Buy the spread
                self.log("BUY SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                self.order_target_size(self.datas[0], self.size)
                self.order_target_size(self.datas[1], -self.hedge_ratio * self.size)
            elif self.zscore > self.upper_bound:
                # Sell the spread
                self.log("SELL SPREAD: A {} B {}".format(self.data_a[0], self.data_b[0]))
                self.order_target_size(self.datas[0], -self.size)
                self.order_target_size(self.datas[1], self.hedge_ratio * self.size)

        # If there is an open trade, wait until the zscore crosses zero
        elif self.position.size > 0 and self.zscore > 0:
            self.log("CLOSE LONG POSITION: A {} B {}".format(self.data_a[0], self.data_b[0]))
            self.order_target_size(self.datas[0], 0)
            self.order_target_size(self.datas[1], 0)

        elif self.position.size < 0 and self.zscore < 0:
            self.log("CLOSE SHORT POSITION: A {} B {}".format(self.data_a[0], self.data_b[0]))
            self.order_target_size(self.datas[0], 0)
            self.order_target_size(self.datas[1], 0)


#%%

if __name__ == "__main__":

    days = 30
    cerebro = bt.Cerebro()

    client = Client(api_key,api_secret, {"verify": path_zert})
    #client.API_URL = 'https://testnet.binance.vision/api'
    data = fetch_crypto_data(10, days, client)

    pairs = find_cointegrated_pairs(data)

    # choose pair wits smallest p-value in pairs
    tickers_pairs = pairs.iloc[0,0:2]
    print(tickers_pairs)

    # get data for pair
    data_df0 = fetch_data(tickers_pairs[0], '1h', str(days * 24), client)
    data_df1 = fetch_data(tickers_pairs[1], '1h', str(days * 24), client)

    data0 = bt.feeds.PandasData(dataname=pd.DataFrame(data_df0))
    data1 = bt.feeds.PandasData(dataname=pd.DataFrame(data_df1))
    cerebro.adddata(data0)
    cerebro.adddata(data1)

    # Add the strategy
    cerebro.addstrategy(PairsTrading)

    # Set the commission and the starting cash
    cerebro.broker.setcommission(commission=0.1)
    cerebro.broker.setcash(100000)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    # Run the backtest
    results = cerebro.run()

    # Print the final portfolio value
    portvalue = cerebro.broker.getvalue()
    print("Final Portfolio Value: ${}".format(round(portvalue, 2)))

    # Print some performance metrics
    print("Sharpe Ratio:", results[0].analyzers.returns.get_analysis()['rnorm'])
    print("Total Trades:", results[0].analyzers.trade_analyzer.get_analysis()['total'])
    print("Winning Trades:", results[0].analyzers.trade_analyzer.get_analysis()['won'])
    print("Losing Trades:", results[0].analyzers.trade_analyzer.get_analysis()['lost'])
    print("Drawdown:", results[0].analyzers.drawdown.get_analysis()['max']['drawdown'])
    print("SQN:", results[0].analyzers.sqn.get_analysis()['sqn'])

#%%









