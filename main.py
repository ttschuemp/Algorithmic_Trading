import pandas as pd
from src.pairs_trading_backtrader import PairTradingStrategy
from src.pairs_trading_functions import find_cointegrated_pairs
from src.load_data import fetch_crypto_data, fetch_data
from binance import Client
from src.api_key_secret import api_key, api_secret, path_zert
import backtrader as bt


if __name__ == "__main__":

    days = 10
    cerebro = bt.Cerebro()

    client = Client(api_key,api_secret, {"verify": path_zert})
    #client.API_URL = 'https://testnet.binance.vision/api'
    data = fetch_crypto_data(10, days, client)

    pairs = find_cointegrated_pairs(data)

    # choose pair wits smallest p-value in pairs
    tickers_pairs = pairs.iloc[0,0:2]

    # get data for pair
    data0 = fetch_data(tickers_pairs[0], '1h', str(days * 24), client)
    data1 = fetch_data(tickers_pairs[1], '1h', str(days * 24), client)



    data0 = bt.feeds.PandasData(dataname=pd.DataFrame(data0))
    data1 = bt.feeds.PandasData(dataname=pd.DataFrame(data1))
    cerebro.adddata(data0)
    cerebro.adddata(data1)

    # Check if data feeds were added
    if len(cerebro.datas) == 2:
        print('Data feeds added successfully')
    else:
        print('Error: Data feeds not added')

    # params
    period=10
#    stake=10
    qty1=0
    qty2=0
    window=30
    std_dev=1
    printout=True
    upper=2.1
    lower=-2.1
    up_medium=0.5
    low_medium=-0.5
    status=0
    portfolio_value=10000
    cash = 10000
    commission = 0.005
    
    #cerebro.addstrategy(MyStrategy)
     # Add the strategy
    cerebro.addstrategy(PairTradingStrategy,
                          period=period,
                          #stake=stake,
                          qty1=qty1,
                          qty2=qty2,
                          window=window,
                          std_dev=std_dev,
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
    #cerebro.plot(volume=False, zdown=False)


#%%

if __name__ == "__main__":

    days = 10
    #cerebro = bt.Cerebro()

    client = Client(api_key,api_secret, {"verify": path_zert})
    #client.API_URL = 'https://testnet.binance.vision/api'
    data = fetch_crypto_data(10, days, client)

    pairs = find_cointegrated_pairs(data)

    # choose pair wits smallest p-value in pairs
    tickers_pairs = pairs.iloc[0,0:2]

    # get data for pair
    data0 = fetch_data(tickers_pairs[0], '1h', str(days * 24), client)
    data1 = fetch_data(tickers_pairs[1], '1h', str(days * 24), client)

    strategy = PairTradingStrategy(period=period,
                                   #stake=stake,
                                   qty1=qty1,
                                   qty2=qty2,
                                   window=window,
                                   std_dev=std_dev,
                                   printout=printout,
                                   upper=upper,
                                   lower=lower,
                                   up_medium=up_medium,
                                   low_medium=low_medium,
                                   status=status,
                                   data0=data0,
                                   data1=data1,
                                   portfolio_value=portfolio_value)
