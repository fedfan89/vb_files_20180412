
# coding: utf-8

# In[533]:


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


# In[534]:


"""OLS model from statsmodels.api as sm
    .df_model # Degrees of Freedom of the Model
    .df_resid; # Degrees of Freedom of the Residuals
    model.endog_names
    model.exog_names

Deprecation Warning:
pandas_new.py:45: FutureWarning: pd.rolling_mean is deprecated for DataFrame and will be removed in a future version, replace with 
    DataFrame.rolling(center=False,window=5).mean()
      rolling_mean = pd.rolling_mean(price_table.iloc[::-1], window = 5).round(2)


Deleted lines of code that may be interesting:
filtered_data = daily_returns[np.isfinite(daily_returns['SPY'])]   .exog_names
price_table_reversed = price_table.iloc[::-1]
rolling_HVs = np.nanstd(daily_returns['SPY'])
rolling_mean = price_table.iloc[::-1].rolling(window=5).mean()
scrubbed.rename(index=str, columns={'AAPL': 'AAPL_scrubbed', 'SPY': 'SPY_scrubbed'}, inplace=True)
"""


# In[535]:


# Import Price Table as Pandas DataFrame from Pickle File for S&P500 + Discretionary Symbols
price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).head(20)[['AAPL', 'SPY']]
price_table.index = pd.to_datetime(daily_returns.index)
daily_returns = price_table / price_table.shift(-1) - 1
model = sm.OLS(daily_returns['AAPL'], daily_returns['SPY'], missing='drop')
results = model.fit()


# In[536]:


# Calculate n-Day Rolling HVs.
rolling_HVs = daily_returns.iloc[::-1].rolling(window=5).std()*math.sqrt(252)


# In[537]:


# Print Statements
# Head(500) returns 498 for results.df_resid. I want that number to be 499.
print("Beta: {:.2f}, Corr.: {:.2f}, n = {:.0f}".format(results.params['SPY'], results.rsquared, results.df_resid))


# In[538]:


#daily_returns['AAPL_predicted'] = daily_returns.SPY*results.params['SPY']


# In[539]:


initial_scrub = daily_returns[(abs(daily_returns.SPY) > .005) & (daily_returns.AAPL < .075)]
#initial_scrub.rename(index=str, columns={'AAPL': 'AAPL_initial_scrub', 'SPY': 'SPY_initial_scrub'}, inplace=True)
model = sm.OLS(initial_scrub['AAPL'], initial_scrub['SPY'], missing='drop')
results = model.fit()
initial_scrub['AAPL_predicted'] = initial_scrub.AAPL*results.params['SPY']
initial_scrub['error'] = initial_scrub.AAPL - initial_scrub.AAPL_predicted
initial_scrub['error_squared'] = initial_scrub.error*initial_scrub.error
initial_scrub


# In[540]:


pct_cutoff = .15
second_scrub = initial_scrub[initial_scrub.error_squared < initial_scrub.error_squared.quantile(1-pct_cutoff)].loc[:, ['AAPL', 'SPY']]
second_scrub


# In[541]:


main = daily_returns.join(initial_scrub)
main

class ScrubParams(object):
    def __init__(self,
                    stock_cutoff: 'float' = None,
                    index_cutoff: 'float' = None,
                    percentile_cutoff: 'float' = None)
        self.stock_cutoff = stock_cutoff
        self.index_cutoff = index_cutoff
        self.percentile_cutoff = percentile.cutoff

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
            initial_scrub = daily_returns[(abs(daily_returns[self.stock]) < self.stock_cutoff]
        else:
            main = daily_returns    
            

        if self.ScrubParams.index_cutoff is not None:
            second_scrub = daily_returns[(abs(daily_returns[self.index]) > self.index_cutoff]
        else:
            main = initial_scrub

        if self.ScrubParams.percentile_cutoff is not None:
            model = sm.OLS(second_scrub[self.stock], second_scrub[self.index], missing='drop')
            results = model.fit()
            second_scrub['y_hat'] = second_scrub[self.stock]*results.params[self.index]
            second_scrub['error'] = second_scrub[self.stock] - second_scrub['y_hat']
            second_scrub['error_squared'] = second_scrub['error']*second_scrub['error']
            
            third_scrub = second_scrub[second_scrub.error_squared < second_scrub.error_squared.quantize(self.ScubParams.percentile_cutoff)].loc[:, [self.stock, self.index]]
            main_scrub = third_scrub
        else:
            main = second_scrub

        model = sm.OLS(main[self.stock], main[self.index], missing='drop')
        results = model.fit()
        return results.params[self.index]
                        









