import backtrader as bt
import backtrader.indicators as btind

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.data0 = self.datas[0]
        self.data1 = self.datas[1]

        # Create an instance of the OLS_TransformationN indicator
        self.transform = btind.OLS_TransformationN(self.data0, self.data1, period=2)

    def next(self):
        # Access the z-score value from the OLS_TransformationN indicator
        zscore = self.transform.zscore[0]
        print('z-score:', zscore)
