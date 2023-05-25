import pandas as pd
from src.pairs_trading_backtrader import PairsTrading
from src.pairs_trading_functions import find_cointegrated_pairs_hurst
from src.load_data import fetch_crypto_data, fetch_data
from binance import Client
from src.api_key_secret import api_key, api_secret #, path_zert
import backtrader as bt
import matplotlib.pyplot as plt
import quantstats
import os


if __name__ == "__main__":
    
    days = 90
    cerebro = bt.Cerebro()

    # Fetch data and find cointegrated pairs
    client = Client(api_key, api_secret)
    data = fetch_crypto_data(50, days, client)
    pairs = find_cointegrated_pairs_hurst(data)

    window = int(pairs['Half Life'][0])
    #window = 150
    std_dev = 1
    size = 0.02

    # Choose the pair with the smallest p-value
    tickers_pairs = pairs.iloc[0, 0:2]
    print(f'trading pair: ' + str(tickers_pairs))

    # Fetch data for the chosen pair
    data_df0 = fetch_data(tickers_pairs[0], '1h', str(days * 24), client)
    data_df1 = fetch_data(tickers_pairs[1], '1h', str(days * 24), client)
    data0 = bt.feeds.PandasData(dataname=pd.DataFrame(data_df0))
    data1 = bt.feeds.PandasData(dataname=pd.DataFrame(data_df1))
    cerebro.adddata(data0)
    cerebro.adddata(data1)

    # Add the strategy
    cerebro.addstrategy(PairsTrading, window=window, std_dev=std_dev, size=size)

    # Set the commission and the starting cash
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.setcash(100000)
    #slippage = 0.001
    #cerebro.broker = btbroker.BackBroker(slip_perc=slippage)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

    # Run the backtest
    results = cerebro.run()

    # Print the final portfolio value
    final_value = cerebro.broker.getvalue()
    print("Final portfolio value: ${}".format(final_value))

    # Get the analyzers and print the results
    trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
    print("Starting cash: ${}".format(cerebro.broker.startingcash))
    print("Ending cash: ${}".format(cerebro.broker.getvalue()))
    print("Total return: {:.2f}%".format(100*((cerebro.broker.getvalue()/cerebro.broker.startingcash)-1)))
    print("Half-Live: " + str(window) + " hours")
    print("Number of trades: {}".format(trade_analyzer.total.closed))
    print("Winning Trades:", results[0].analyzers.trade_analyzer.get_analysis()['won']['total'])
    print("Losing Trades:", results[0].analyzers.trade_analyzer.get_analysis()['lost']['total'])
    print("Win Ratio:", results[0].analyzers.trade_analyzer.get_analysis()['won']['total'] /
          trade_analyzer.total.closed)


    # Get the strategy instance
    strategy_instance = results[0]

    # Plot the spread, zscore, and hedge ratio
    plt.subplot(4, 1, 1)
    plt.plot(strategy_instance.spread_history_full)
    plt.title(f'Spread {list(tickers_pairs)}')

    plt.subplot(4, 1, 2)
    plt.plot(strategy_instance.zscore_history)
    plt.axhline(strategy_instance.upper_bound, color='r')
    plt.axhline(strategy_instance.lower_bound, color='r')
    plt.title("Z-score")
    plt.legend(["Z-score"])

    plt.subplot(4, 1, 3)
    plt.plot(strategy_instance.hedge_ratio_history)
    plt.title("Hedge ratio")
    plt.legend(["Hedge ratio"])

    plt.subplot(4, 1, 4)
    plt.plot(strategy_instance.hurst_history_2)
    plt.title("Hurst")
    plt.legend(["Hurst"])

    plt.tight_layout()
    plt.savefig('strategy')
    plt.show()

    # create cerebro chart
    cerebro.plot(iplot=True, volume=False)

    # create quantstats charts & statistics html
    portfolio_stats = strategy_instance.analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)

    quantstats.reports.html(returns, output='stats.html', title='Backtrade Pairs')

