import pandas as pd


# make a function to load data
def load_data(path):
    # load csv file in data pairs
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    # rename column index to date
    data.index.name = 'date'
    return data

