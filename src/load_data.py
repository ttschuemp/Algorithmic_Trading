import pandas as pd


# load csv file in data pairs
data = pd.read_csv('src/data/data pairs.csv', index_col=0, parse_dates=True)
print(data.head())

