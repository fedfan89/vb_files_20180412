from paul_resources import PriceTable, daily_returns
import numpy as np
import pandas as pd
from operator import mul

price_table = PriceTable
returns = daily_return(PriceTable)

def alpha_df(df: 'df of returns', lookback):
    stocks = df.columns.values
    returns = []
    HVs = []
    ratios = []
    adj_ratios = []
    for stock in stocks:
        df2 = df + 1
        total_move = reduce(mul, df2, 1) - 1
        returns.append[total_move]
        print(df2, total_move)
        
        df = df[stock].dropna(axis=0, how='any').head(lookback)
        HV = df.std(ddof=0)*math.sqrt(252)
        HVs.append(HV)

        ratio = total_move / HV
        ratios.append(ratio)

        adj_ratio = total_move*math.sqrt(252/lookback) / HV
        adj_ratios.append(adj_ratio)
    df = pd.DataFrame(list(zip(stocks, returns, HVs, ratios, adj_ratios)))
   return df

df = alpha_df(returns)
print(df)
