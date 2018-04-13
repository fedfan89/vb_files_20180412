import numpy as np
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
"""


# Import Price Table as Pandas DataFrame from Pickle File for S&P500 + Discretionary Symbols
price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).head(50)[['AAPL', 'SPY']]
daily_returns = price_table / price_table.shift(-1) - 1
model = sm.OLS(daily_returns['AAPL'], daily_returns['SPY'], missing='drop')
results = model.fit()

# Calculate n-Day Rolling HVs.
rolling_HVs = daily_returns.iloc[::-1].rolling(window=5).std()*math.sqrt(252)



# Print Statements
# Head(500) returns 498 for results.df_resid. I want that number to be 499.
print("Beta: {:.2f}, Corr.: {:.2f}, n = {:.0f}".format(results.params['SPY'], results.rsquared, results.df_resid))

print(rolling_HVs.round(3))
print(rolling_HVs)
