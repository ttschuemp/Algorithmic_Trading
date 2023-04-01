import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (
        ('period', 10),
        ('devfactor', 2),
    )

    def __init__(self, data1, data2):
        self.data = MyDataFeed(dataname='my_data.csv')
        self.data1 = self.datas[0]
        self.data2 = self.datas[1]
        self.zscore = bt.indicators.ZScore(self.data1 - self.data2, period=self.params.period, devfactor=self.params.devfactor)

    def next(self):
        if self.zscore[0] > 1.0:
            self.sell(data=self.data1)
            self.buy(data=self.data2)
        elif self.zscore[0] < -1.0:
            self.sell(data=self.data2)
            self.buy(data=self.data1)