import pandas as pd
from src.pairs_trading_backtrader import PairTradingStrategy
from src.pairs_trading_functions import find_cointegrated_pairs
from src.load_data import fetch_crypto_data
from binance import Client
from api_key_secret import api_key, api_secret
import backtrader as bt

if __name__ == "__main__":

    cerebro = bt.Cerebro()

    client = Client(api_key,api_secret)
    client.API_URL = 'https://testnet.binance.vision/api'
    data = fetch_crypto_data(10, 2, client)

    pairs = find_cointegrated_pairs(data)

    # choose pair wits smallest p-value in pairs
    tickers_pairs = pairs.iloc[0,0:2]


    # new data frame with only the two tickers
    data_pairs = data[tickers_pairs]

    data0 = bt.feeds.PandasData(dataname=pd.DataFrame(data_pairs.iloc[:, 0]))
    data1 = bt.feeds.PandasData(dataname=pd.DataFrame(data_pairs.iloc[:, 1]))
    cerebro.adddata(data0)
    cerebro.adddata(data1)

    # params
    period=10
#    stake=10
    qty1=0
    qty2=0
    printout=True
    upper=2.1
    lower=-2.1
    up_medium=0.5
    low_medium=-0.5
    status=0
    portfolio_value=10000
    cash = 10000
    commission = 0.005

     # Add the strategy
    cerebro.addstrategy(PairTradingStrategy,
                        period=period,
                        #stake=stake,
                        qty1=qty1,
                        qty2=qty2,
                        printout=printout,
                        upper=upper,
                        lower=lower,
                        up_medium=up_medium,
                        low_medium=low_medium,
                        status=status,
                        #data0=data0,
                        #data1=data1,
                        portfolio_value=portfolio_value)

    # Add the commission - only stocks like a for each operation
    cerebro.broker.setcash(cash)

    # Add the commission - only stocks like a for each operation
    cerebro.broker.setcommission(commission=commission)

    # And run it
    cerebro.run(runonce=False,
                preload=True,
                oldsync=True)
    
    # Plot if requested
    cerebro.plot(volume=False, zdown=False)
