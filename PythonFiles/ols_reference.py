import pickle
import copy
import pprint
import decimal
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from time_decorator import my_time_decorator
import math
import matplotlib.pyplot as plt
import scipy.stats as ss


#Take a list of (x,y) pairs and return a list of tuples containing (x, y, y_hat, error, error_squared)
#Along the way, calculate correlation (corr), beta (beta1), alpha (beta0), error, and error_squared
def ols_params(pairs: 'list of (x,y) tuples' = [], name = 'default') -> 'ols information as a list of tuples':
    x, y = [i[0] for i in pairs], [i[1] for i in pairs]
    x_bar, y_bar = (sum(x)/len(x)), (sum(y)/len(y))
    beta1 = sum([(i[1] - y_bar)*(i[0] - x_bar) for i in pairs]) / sum([(i[0] - x_bar)**2 for i in pairs])
    beta0 = y_bar - beta1*x_bar
    corr = np.corrcoef(x, y)[1,0]
    
    #calculate y_hat, error, error_squared
    y_hat = [round(beta0 + beta1*i, 4) for i in x]
    error = [round(y[i] - y_hat[i], 4) for i in range(len(y))]
    error_squared = [round(i**2, 5) for i in error]
    
    #Print Statement
    print(name, ": ",
            ", Beta = ", round(beta1, 2),
            ", Corr. = ", round(corr, 2),
            ", n = ", len(x),
            sep="")

    #return a list of tuples containing (x, y, y_hat, error, error_squared)
    return list(zip(x, y, y_hat, error, error_squared))

def ols_beta(pairs: 'list of (x,y) tuples' = []) -> 'float':
    x, y = [i[0] for i in pairs], [i[1] for i in pairs]
    x_bar, y_bar = (sum(x)/len(x)), (sum(y)/len(y))
    return round(sum([(i[1] - y_bar)*(i[0] - x_bar) for i in pairs]) / sum([(i[0] - x_bar)**2 for i in pairs]), 2)

def ols_beta2(pairs: 'list of (x,y) tuples' = []) -> 'float':
    x, y = [i[0] for i in pairs], [i[1] for i in pairs]
    x_bar, y_bar = (sum(x)/len(x)), (sum(y)/len(y))
    return round(sum([(i[1] - y_bar)*(i[0] - x_bar) for i in pairs]) / sum([(i[0] - x_bar)**2 for i in pairs]), 2)

a = [.02, .03, .04, .01]
b = [.05, .01, .02, .06]
c = list(zip(a,b))
#ols_params(c, name = "OLS Example")
d = ols_beta(c)
#print("Beta: ", d, sep="")

#Take a list of (x,y) pairs and return a list of tuples containing (x, y, y_hat, error, error_squared)
#Along the way, calculate correlation (corr), beta (beta1), alpha (beta0), error, and error_squared

class OLS(object):
    def __init__(self,
            pairs: 'list of (x,y) tuples' = [],
            name = 'OLS Default') -> 'ols information as a list of tuples':
        self.name = name
        self.x = [i[0] for i in pairs]
        self.y = [i[1] for i in pairs]
        self.x_bar = (sum(self.x)/len(self.x))
        self.y_bar = (sum(self.y)/len(self.y))
        self.beta1 = sum([(i[1] - self.y_bar)*(i[0] - self.x_bar) for i in pairs]) / sum([(i[0] - self.x_bar)**2 for i in pairs])
        self.beta0 = self.y_bar - self.beta1*self.x_bar
        self.corr = np.corrcoef(self.x, self.y)[1,0]
        
        #calculate y_hat, error, error_squared
        self.y_hat = [round(self.beta0 + self.beta1*i, 4) for i in self.x]
        self.error = [round(self.y[i] - self.y_hat[i], 4) for i in range(len(self.y))]
        self.error_squared = [round(i**2, 5) for i in self.error]
    
    def summary(self):
        #Print Statement
        print(self.name, ": ",
                ", Beta = ", round(self.beta1, 2),
                ", Corr. = ", round(self.corr, 2),
                ", n = ", len(self.x),
                sep="")

    #return a list of tuples containing (x, y, y_hat, error, error_squared)
    #return list(zip(x, y, y_hat, error, error_squared))

#print(OLS(c).corr)
#OLS(c).summary()
