from paul_resources import PriceTable, daily_returns
import numpy as np
import pandas as pd
import math
from operator import mul
from functools import reduce
import matplotlib.pyplot as plt

def alpha_df(df: 'df of prices', lookback):
    prices_df = df.head(lookback)
    returns_df = daily_returns(prices_df).dropna(axis=0, how='any')
    #print(prices_df.shape, returns_df.shape)
    #print(prices_df)
    #print(prices_df.isnull())
    #print(prices_df.isnull().values.ravel().sum())
    #print(prices_df.isnull().values.ravel())
    #print(returns_df)

    stocks = df.columns.values.tolist()
    #for stock in stocks:
    #    result = prices_df[prices_df[stock].isnull()].index.tolist()
    #    print(stock, ": ", result, sep='')

    returns = []
    HVs = []
    ratios = []
    adj_ratios = []
    
    for stock in stocks:
        total_move = prices_df[stock].head(1).iloc[0] / prices_df[stock].tail(1).iloc[0] - 1
        returns.append(total_move)
        #print(returns_df.shape)
        
        HV = returns_df[stock].std(ddof=0)*math.sqrt(252)
        HVs.append(HV)
        if stock == 'ALGN':
            print(HV)

        ratio = total_move / HV
        ratios.append(ratio)

        adj_ratio = total_move*math.sqrt(252/lookback) / HV
        adj_ratios.append(adj_ratio)

    alpha_df = pd.DataFrame(list(zip(stocks, returns, HVs, ratios, adj_ratios))).round(3)
    alpha_df.rename(index=str, columns={0: 'Stock', 1: 'Return', 2: 'HV', 3: 'Ratio', 4: 'Adj_Ratio'}, inplace=True)
    alpha_df.set_index('Stock', inplace=True)
    return alpha_df

stocks = [i for i in PriceTable.columns.values.tolist() if i not in {'BHF', 'CBRE', 'FTV', 'WELL', 'BKNG'}]
stocks2 = ['NVDA', 'ALGN', 'NKTR']
stocks3 = ['WELL']

alpha_df = alpha_df(PriceTable[stocks], 252)
print(alpha_df.sort_values('Adj_Ratio', ascending=False))
#print(alpha_df.sort_index(ascending=True))
#print(PriceTable[stocks2])
#print(PriceTable['BKNG'])
print(alpha_df.shape)

def graph():
    values = alpha_df['Adj_Ratio'].tolist()
    bins = np.arange(-5.5, 6.5, 1)
    print(bins)
    plt.hist(values, bins, histtype = 'bar', rwidth=.8)
    plt.xlabel('Adj_Ratio')
    plt.ylabel('Frequency')
    plt.title('S&P 500 Alpha Distribution')
    plt.legend()
    plt.show()

graph()
