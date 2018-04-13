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
from statistics import mean
from operator import itemgetter

#Take a tuple of information
#Create an object that contains the information
class OLSPoint(object):
    def __init__(self, item):
        self.contents = item
        self.x = item[0]
        self.y = item[1]
        self.date = item[2]
        self.y_hat = item[3]
        self.error = item[4]
        self.error_squared = item[5]

#Take a list of (x,y) pairs.
#Calculate beta (beta1), alpha (beta0), and correlation (corr)
#Calculate y_hat, error, error_squared terms for each pair
class OLS(object):
    def __init__(self,
            pairs: 'list of (x,y) tuples' = [],
            name = 'OLS Default') -> 'ols information as a list of tuples':
        self.pairs = pairs
        self.name = name
        self.count = len(pairs)
        self.x = [i[0] for i in pairs]
        self.y = [i[1] for i in pairs]
        self.dates = [i[2] for i in pairs]
        #self.x = list(map(itemgetter(0), pairs))
        #self.y = list(map(itemgetter(1), pairs))
        #self.dates = list(map(itemgetter(2), pairs))
        
        self.core_calculations()
        
        #List of tuples containing (x, y, y_hat, error, error_squared)
        self.contents = [list(i) for i in zip(self.x, self.y, self.dates, self.y_hat, self.error, self.error_squared)]
        self.olspoints = [OLSPoint(i) for i in self.contents]

    def core_calculations(self):
        """Performs the core calculations for the OLS Regression"""
        self.x_bar = mean(self.x)
        self.y_bar = mean(self.y)
        self.beta1 = sum([(i[1] - self.y_bar)*(i[0] - self.x_bar) for i in self.pairs]) / sum([(i[0] - self.x_bar)**2 for i in self.pairs])
        self.beta0 = self.y_bar - self.beta1*self.x_bar
        self.corr = np.corrcoef(self.x, self.y)[1,0]
        
        #Calculate y_hat, error, error_squared
        self.y_hat = [round(self.beta0 + self.beta1*i, 4) for i in self.x]
        self.error = [round(self.y[i] - self.y_hat[i], 4) for i in range(len(self.y))]
        self.error_squared = [round(i**2, 5) for i in self.error]
    
    def summary(self):
        """Print Statement that outputs Beta, Corr., and n"""
        message = "{}: Beta = {}, Corr. = {}, n = {}".format(self.name,
                                                                round(self.beta1, 2), 
                                                                round(self.corr, 2), 
                                                                len(self.x))
        print(message)

        #F_String (not working)
        #message = f"{self.name}: Beta = {round(self.beta1, 2)}, Corr. = {round(self.corr, 2)}, n = {len(self.x)}"

    def ols(self):
        """Return a list of tuples containing (x, y, y_hat, error, error_squared)"""
        return list(zip(self.x, self.y, self.dates, self.y_hat, self.error, self.error_squared))


