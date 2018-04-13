import numpy as np
import datetime as dt
import pandas as pd
import pickle
import copy
import pprint
import decimal
import statsmodels.formula.api as sm
from time_decorator import my_time_decorator
from ols import OLS
import math
import matplotlib.pyplot as plt
import scipy.stats as ss
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from ols2 import OLS as MainOLS

PriceTable = pickle.load(open('sp500_price_table.pkl', 'rb'))
PriceTable.index = pd.to_datetime(PriceTable.index)

InformationTable = pd.read_csv('information_table.csv')
InformationTable.rename(columns = {'Last Close': 'Price', 'Ticker': 'Stock', 'Market Cap ': 'Market Cap'}, inplace=True)
InformationTable.set_index('Stock', inplace=True)

def daily_returns(price_table: 'df of prices') -> 'df of daily_returns':
    return price_table / price_table.shift(-1) - 1

def tprint(*args):
    print("TPrint Here--------")
    for arg in args:
        print("Type: ", type(arg), "\n", "Obj: ", arg, sep='')

def rprint(*args):
    print("RPrint Here-------")
    for arg in args:
        if args.index(arg) == len(args)-1:
            e = " "
        else:
            e = ", "
        if type(arg) is float:
            print(round(arg, 3), end = e)
        else:
            print(arg, end = e)

class Aaron(object):
    pass

if __name__ == "__main__":
    print(PriceTable.index.values)
    print(PriceTable.index)
    print(type(PriceTable.index.values))
    print(type(PriceTable.index))
    print(PriceTable.index.values.tolist())
    print(PriceTable.index.tolist())
    print(type(PriceTable.index.values.tolist()[0]))
    print(type(PriceTable.index.tolist()[0]))
    
    tprint(InformationTable.columns)
