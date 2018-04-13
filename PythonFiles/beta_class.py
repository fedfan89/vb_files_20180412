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

    def beta(self):
        price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).head(20)[[self.stock, self.index]]
        price_table.index = pd.to_datetime(price_table.index)
        daily_returns = price_table / price_table.shift(-1) - 1

        if self.ScrubParams.stock_cutoff is not None:
            initial_scrub = daily_returns[abs(daily_returns[self.stock]) < self.ScrubParams.stock_cutoff]
        else:
            main = daily_returns    
            

        if self.ScrubParams.index_cutoff is not None:
            second_scrub = daily_returns[abs(daily_returns[self.index]) > self.ScrubParams.index_cutoff]
        else:
            main = initial_scrub

        if self.ScrubParams.percentile_cutoff is not None:
            model = sm.OLS(second_scrub[self.stock], second_scrub[self.index], missing='drop')
            results = model.fit()
            second_scrub['y_hat'] = second_scrub[self.stock]*results.params[self.index]
            second_scrub['error'] = second_scrub[self.stock] - second_scrub['y_hat']
            second_scrub['error_squared'] = second_scrub['error']*second_scrub['error']
            
            third_scrub = second_scrub[second_scrub.error_squared < second_scrub.error_squared.quantile(self.ScrubParams.percentile_cutoff)].loc[:, [self.stock, self.index]]
            main = third_scrub
        else:
            main = second_scrub

        model = sm.OLS(main[self.stock], main[self.index], missing='drop')
        results = model.fit()
        return results.params[self.index]
                        
    def describe(self):
        """results.rsquared, results.df_resid"""
        print("Beta: {:.2f}, Corr.: {:.2f}, n = {:.0f}".format(self.beta(), .99, 498))
