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
import matplotlib.ticker as tkr
import scipy.stats as ss
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from ols2 import OLS as MainOLS
from paul_resources import PriceTable
from paul_resources import daily_returns


def OLS_df(df: 'DataFrame of (stock, index) daily returns') -> 'DataFrame of (stock, index) daily returns with y_hat, error, error_squared':
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
        self.price_table = PriceTable.head(self.lookback)[[self.stock, self.index]]
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
    def __init__(self, stock: 'str', lookback: 'int', base = None):
        self.stock = stock
        self.lookback = lookback
        if base is None:
            self.base = self.stock
        else:
            self.base = base

        self.price_table = PriceTable.head(self.lookback)[[self.stock]]
        self.daily_returns = daily_returns(self.price_table).head(self.lookback)

    @property
    def stock_base_price(self):
        return self.price_table.tail(1)[self.stock].iloc[0]
        
    @property   
    def base_price(self):
        if self.base == self.stock:
            return self.price_table.tail(1)[self.stock].iloc[0]
        elif type(self.base) is str:
            return PriceTable.head(self.lookback).tail(1)[self.base].iloc[0]
        elif type(self.base) == float or type(self.base == int):
            return self.base
        else:
            raise ValueError
    
    @property
    def prices_df(self):
        mult = self.base_price / self.stock_base_price
        return self.price_table.iloc[::-1] * mult
    
    @property
    def chart_name(self):
        if self.stock == self.base:
            return "{}".format(self.stock)
        else:
            return "{} (Base: {})".format(self.stock, self.base)  
    
    def stock_line(self, color, name = None):
        if name is None:
            name = self.chart_name
        return (self.prices_df, color, name)

class StockLineBetaAdjusted(StockLineSimple):
    def __init__(self, stock: 'str', lookback: 'int', beta: 'float', index: 'str', base = None):
        super().__init__(stock, lookback, base)
        self.beta = beta
        self.index = index

    @property   
    def adjusted_returns(self):
        index_prices = PriceTable.head(self.lookback)[self.index]
        index_returns = daily_returns(index_prices).to_frame()
        df = (1 + self.daily_returns[self.stock]) / (1 + index_returns[self.index]*self.beta) - 1
        df.name = 'Adj. Returns'
        return df

    @property
    def prices_df(self):
        dates = list(reversed(self.adjusted_returns.index.values))
        returns = list(reversed(self.adjusted_returns.tolist()))
        prices = [self.base_price]
        for i in range(1, len(dates)):
            daily_move = returns[i]
            prices.append((1 + daily_move)*prices[i-1])
        df = pd.DataFrame({'dates': dates, 'Adj_Prices': prices})
        df.set_index('dates', inplace=True)
        return df

    @property
    def chart_name(self):
        return "{}; Index: {}, Beta: {:.2}".format(self.stock, self.index, self.beta)


class StockChart(object):
    def __init__(self, stock_lines: 'list of stock lines'):
        self.stock_lines = stock_lines

    def run(self):
        fig = plt.figure()
        ax1 = plt.subplot2grid((1,1), (0,0)) #ax1 is a subplot, ax2 would be a second subplot
        ax1.grid(True, color='gray', linestyle='-', linewidth=.5)
        
        for stock_line in self.stock_lines:
            prices_df = stock_line[0]
            c = stock_line[1]
            l = stock_line[2]
            x = prices_df.index.tolist()
            y = prices_df.iloc[:, 0].tolist()
            ax1.plot(x, y, color = c, label = l)
        
        
        plt.title("Paul's Fancy Stock Chart\nCheck It Out")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.style.use('ggplot')
        plt.legend()
        plt.subplots_adjust(left=.09, bottom=.12, right =.94, top=.9, wspace=.2, hspace=0)
        
        plt.show()

stock = 'SRPT'
stock2 = 'AAPL'
index = 'XBI'
lookback = 30

base = 100
beta = Beta(stock, index, lookback, ScrubParams(.075, .0125, .8)).beta
beta2 = Beta(stock2, index, lookback, ScrubParams(.075, .0125, .8)).beta
#beta, beta2 = 0.0, 0.0

# stock_lines to plot
stock_line = StockLineSimple(stock, lookback, base)
index_line = StockLineSimple(index, lookback, base)
stock_line_adj = StockLineBetaAdjusted(stock, lookback, beta, index, base)
stock_line_adj2 = StockLineBetaAdjusted(stock2, lookback, beta2, index, base)

stock_lines = [
                stock_line.stock_line(color = 'red'),
                index_line.stock_line(color = 'black'),
                stock_line_adj.stock_line(color = 'blue'),
                #stock_line_adj2.stock_line(color = 'c')
                ]


StockChart(stock_lines).run()














"""
Create Adjusted Stock Chart based on beta to index -> Line Graphs
        if base100 == True:
             plt.plot(self.dates_reversed, self.raw_index_prices_base100, color='black', label=(str(self.index) + " (base100)"))
            plt.plot(self.dates_reversed, self.raw_stock_prices_base100, color='red', label = (str(self.stock) + " (base100)"))
            plt.plot(self.dates_reversed, self.adj_stock_prices_base100, color='gold', label= (str(self.stock) + " beta-adjusted (base100)"))
        else:
            plt.plot(self.dates_reversed, self.raw_stock_prices, color='red', label=str(self.stock)+ " Raw Prices")
            plt.plot(self.dates_reversed, self.raw_index_prices_baseStock, color='black', label=str(self.index) + " (base: " + str(self.index) + ")")
            plt.plot(self.dates_reversed, self.adj_stock_prices, color='gold', label=str(self.stock) + " Adj. Prices based on Beta to " + str(self.stock) + "")
        plt.title(str(self.stock) + " Stock Chart" + "; Index: " + str(self.index) + "; Beta Est. = " +str(round(self.beta_graph,2)))                                       plt.xlabel("Date")
    plt.ylabel("Stock Price")
        plt.xticks(rotation=45)
        plt.style.use('ggplot')
        plt.legend()
        plt.show()
"""

