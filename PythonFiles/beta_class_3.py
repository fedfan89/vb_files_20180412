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
from paul_resources import PriceTable
from paul_resources import daily_returns


def OLS_df(df: 'DataFrame of daily returns)') -> 'DataFrame of daily returns with y_hat, error, error_squared':
    df = df
    stock = df.columns[0]
    index = df.columns[1]
    model = sm.OLS(df[stock], df[index], missing = 'drop')
    results = model.fit()

    df['y_hat'] = df[index]*results.params[index]
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
        
        #self.price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).head(self.lookback)[[self.stock, self.index]]
        self.price_table = My_Data.PriceTable.head(self.lookback)[[self.stock, self.index]]
        #self.daily_returns = self.price_table / self.price_table.shift(-1) - 1
        self.daily_returns = daily_returns(self.price_table)

    @property
    def initial_scrub(self):
        if self.ScrubParams.stock_cutoff:
            return self.daily_returns[abs(self.daily_returns[self.stock]) <= self.ScrubParams.stock_cutoff]
        else:
            return None
    
    @property
    def second_scrub(self):
        if self.ScrubParams.stock_cutoff and self.ScrubParams.index_cutoff:
            df = self.initial_scrub[abs(self.initial_scrub[self.index]) >= self.ScrubParams.index_cutoff]
            df = OLS_df(df)
            return df
        else:
            return None

    @property
    def third_scrub(self):
        if self.ScrubParams.stock_cutoff and self.ScrubParams.index_cutoff and self.ScrubParams.percentile_cutoff:
            cutoff = self.second_scrub['error_squared'].quantile(self.ScrubParams.percentile_cutoff)
            return self.second_scrub[self.second_scrub['error_squared'] < cutoff].loc[:, [self.stock, self.index]]
        else:
            return None
    
    @property
    def main(self):
        if self.ScrubParams.percentile_cutoff:
            return self.third_scrub
        elif self.ScrubParams.index_cutoff:
            return self.second_scrub
        elif self.ScrubParams.stock_cutoff:
            return self.initial_scrub
        else:
            return self.daily_returns

    @property
    def OLS_model_results(self):
        model = sm.OLS(self.main[self.stock], self.main[self.index], missing='drop')
        results = model.fit()
        return results
    
    @property
    def OLS_object(self):
        stock_returns = list(self.main[self.stock])
        index_returns = list(self.main[self.index])
        dates = list(self.main.index.values)
        return MainOLS(list(zip(index_returns, stock_returns, dates)))

    @property
    def beta(self):
        return self.OLS_model_results.params[self.index]

    @property
    def beta1(self):
        return self.OLS_object.beta1
    
    @property
    def corr(self):
        return self.OLS_object.corr

    @property
    def rsquared(self):
        return self.OLS_model_results.rsquared

    @property
    def degrees_of_freedom(self):
        return self.OLS_model_results.df_resid

    @property
    def scrub_type(self):
        if self.ScrubParams.percentile_cutoff:
            return "Third_Scrub"
        elif self.ScrubParams.index_cutoff:
            return "Second_Scrub"
        elif self.ScrubParams.stock_cutoff:
            return "Initial_Scrub"
        else:
            return "Raw_Returns"

    def describe(self):
        """results.rsquared, results.df_resid"""
        #print("{}({}, {}, {})".format(self.scrub_type, self.ScrubParams.stock_cutoff, self.ScrubParams.index_cutoff, self.ScrubParams.percentile_cutoff))
        print("{}-> Beta: {:.2f}, Corr.: {:.2f}, n = {:.0f}".format(self.scrub_type, self.beta1, self.corr, self.degrees_of_freedom))
        
        #print(self.ScrubParams.stock_cutoff, self.ScrubParams.index_cutoff, self.ScrubParams.percentile_cutoff)
        #print(type(self.ScrubParams.stock_cutoff), type(self.ScrubParams.index_cutoff), type(self.ScrubParams.percentile_cutoff))

        # TypeError: non-empty format string passed to object.__format__
        #print("{}({:.2f}, {:.2f}, {:.2f})-> Beta: {:.2f}, Corr.: {:.2f}, n = {:.0f}".format(self.scrub_type, self.ScrubParams.stock_cutoff, self.ScrubParams.index_cutoff, self.ScrubParams.percentile_cutoff, self.beta1, self.corr, self.degrees_of_freedom))
    
    def describe2(self):
        """results.rsquared, results.df_resid"""
        print("Beta: {:.2f}, Corr.: {:.2f}, n = {:.0f}".format(self.beta, math.sqrt(self.rsquared), self.degrees_of_freedom))

    def scrub_trajectory(self):
        """Here I show a print statement for the Beta calculation at each step of the scrubbing process: Raw_Returns, Initial_Scrub, Second_Scrub, Third_Scrub"""
        print(self.stock, self.index, self.lookback, self.ScrubParams, self.ScrubParams.stock_cutoff, self.ScrubParams.index_cutoff, self.ScrubParams.percentile_cutoff)
        Beta(self.stock, self.index, self.lookback, ScrubParams()).describe()
        Beta(self.stock, self.index, self.lookback, ScrubParams(2, self.ScrubParams.index_cutoff, 1.0)).describe()
        Beta(self.stock, self.index, self.lookback, ScrubParams(self.ScrubParams.index_cutoff, self.ScrubParams.index_cutoff, 1.0)).describe()
        Beta(self.stock, self.index, self.lookback, ScrubParams(self.ScrubParams.index_cutoff, self.ScrubParams.index_cutoff, self.ScrubParams.percentile_cutoff)).describe()





class StockLineSimple(object):
    def __init__(self, stock: 'str', lookback: 'int'):
        self.stock = stock
        self.lookback = lookback

        self.price_table = PriceTable.head(self.lookback)[[self.stock]]
        self.daily_returns = daily_returns(self.price_table).head(self.lookback)

    @property   
    def base_price(self):
        return self.price_table.tail(1)[self.stock].iloc[0]

    @property
    def stock_line(self):
        return self.price_table.iloc[::-1]

class StockLineComplex(StockLineSimple):
    def __init__(self, stock: 'str', lookback: 'int', 
                    index: 'str',
                    beta: 'float',
                    to_graph: 'str',
                    base: 'str'):
        self.stock = stock
        self.index = index
        self.beta = beta
        self.to_graph = to_graph
        self.base = base
        self.lookback = lookback

        self.price_table = PriceTable.head(self.lookback)[[self.stock, self.index]]
        self.daily_returns = daily_returns(self.price_table).head(self.lookback)

        self.daily_returns['adj_returns'] = (1 + self.daily_returns[self.stock]) / (1 + self.daily_returns[self.index]*self.beta) -1
        print(self.daily_returns.head(5))


    @property   
    def base_price(self):
        if self.base == self.stock:
            return self.price_table.tail(1)[self.stock].iloc[0]
        elif self.base == self.index:
            return self.price_table.tail(1)[self.index].iloc[0]
        elif type(self.base) == float or type(self.base == int):
            return self.base
        else:
            raise ValueError

    def stock_line(self):
        pass
        #self.adj_returns = (1 + self.daily_returns[self.stock]) / (1 + self.daily_returns[self.index]*self.beta) -1
        #print(self.adj_returns.head(5))



class StockLine(object):
    def __init__(self,
                    stock: 'str',
                    index: 'str',
                    beta: 'float',
                    to_graph: 'str',
                    base: 'str',
                    lookback: 'int'):
        self.stock = stock
        self.index = index
        self.beta = beta
        self.to_graph = to_graph
        self.base = base
        self.lookback = lookback

        self.price_table = PriceTable.head(self.lookback)[[self.stock, self.index]]
        self.daily_returns = daily_returns(self.price_table).head(self.lookback)

        self.daily_returns['adj_returns'] = (1 + self.daily_returns[self.stock]) / (1 + self.daily_returns[self.index]*self.beta) -1
        print(self.daily_returns.head(5))


    @property   
    def base_price(self):
        if self.base == self.stock:
            return self.price_table.tail(1)[self.stock].iloc[0]
        elif self.base == self.index:
            return self.price_table.tail(1)[self.index].iloc[0]
        elif type(self.base) == float or type(self.base == int):
            return self.base
        else:
            raise ValueError

    def stock_line(self):
        pass
        #self.adj_returns = (1 + self.daily_returns[self.stock]) / (1 + self.daily_returns[self.index]*self.beta) -1
        #print(self.adj_returns.head(5))



StockLine('SRPT', 'XBI', 5.0, 'SRPT', 'SRPT', 20)

srpt = StockLineSimple('SRPT', 50)
print(srpt.base_price)
print(srpt.stock_line)
