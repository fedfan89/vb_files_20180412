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

def OLS(df: 'DataFrame of resturns)') -> 'DataFrame of returns with y_hat, error, error_squared':
    df = df
    stock = df.columns[0]
    index = df.columns[1]
    model = sm.OLS(df[stock], df[index], missing = 'drop')
    results = model.fit()

    df['y_hat'] = df[stock]*results.params[index]
    df['error'] = df[stock] - df['y_hat']
    df['error_squared'] = df['error']*df['error']
    return df

class ScrubParams(object):
    def __init__(self,
                    stock_cutoff: 'float' = None,
                    index_cutoff: 'float' = None,
                    percentile_cutoff: 'float' = None):
        self.stock_cutoff = stock_cutoff
        self.index_cutoff = index_cutoff
        self.percentile_cutoff = percentile_cutoff

class Beta(object):
    def __init__(self, stock: 'str', index: 'str', lookback: 'int', ScrubParams: 'obj'):
        self.stock = stock
        self.index = index
        self.lookback = lookback
        self.ScrubParams = ScrubParams
        
        self.price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).head(self.lookback)[[self.stock, self.index]]
        self.price_table.index = pd.to_datetime(self.price_table.index)
        self.daily_returns = self.price_table / self.price_table.shift(-1) - 1

    @property
    def initial_scrub(self):
        if self.ScrubParams.stock_cutoff is not None:
            df = self.daily_returns[abs(self.daily_returns[self.stock]) < self.ScrubParams.stock_cutoff]
            print("Initial Scrub&&&&&&&&&&&&&&&&&&&&&", "\n", df)
            return df
        else:
            return None
    
    @property
    def second_scrub(self):
        if self.ScrubParams.stock_cutoff is not None and self.ScrubParams.index_cutoff is not None:
            df = self.initial_scrub[abs(self.initial_scrub[self.index]) > self.ScrubParams.index_cutoff]
            df = OLS(df)
            print("Second Scrub=========================", "\n", df)
            return df
        else:
            return None

    @property
    def third_scrub(self):
        if self.ScrubParams.stock_cutoff is not None and self.ScrubParams.index_cutoff is not None and self.ScrubParams.percentile_cutoff is not None:
            df = self.second_scrub[self.second_scrub.error_squared < self.second_scrub.error_squared.quantile(self.ScrubParams.percentile_cutoff)].loc[:, [self.stock, self.index]]
            print("Third Scrub --------------------------", "\n", df)
            return df
        else:
            return None
    
    @property
    def main(self):
        if self.ScrubParams.percentile_cutoff is not None:
            return self.third_scrub
        elif self.ScrubParams.index_cutoff is not None:
            return self.second_scrub
        elif self.ScrubParams.stock_cutoff is not None:
            return self.initial_scrub
        else:
            return self.daily_returns

    @property
    def beta(self):
        model = sm.OLS(self.main[self.stock], self.main[self.index], missing='drop')
        results = model.fit()
        return results.params[self.index]
                        
    def describe(self):
        """results.rsquared, results.df_resid"""
        print("Beta: {:.2f}, Corr.: {:.2f}, n = {:.0f}".format(self.beta(), .99, 498))
