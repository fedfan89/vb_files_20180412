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

class OLSPoint(object):
    """Take a tuple containing OLS information. Create an OLS object that contains that information."""
    def __init__(self, item):
        self.contents = item
        self.x = item[0]
        self.y = item[1]
        self.date = item[2]
        self.y_hat = item[3]
        self.error = item[4]
        self.error_squared = item[5]

    def __repr__(self):
        return 'OLS Point: ({:.2f}%, {:.2f}%), {}'.format(self.x*100, self.y*100, self.date)

class OLS(object):
    """Take a list of (x,y, date) pairs. The OLS Object contains OLS information about the pairing.
        --Beta (beta1), alpha (beta0), and correlation (corr)
        --y_hat, error, error_squared terms for each pair
    """
    def __init__(self,
            pairs: 'list of (x,y, date) tuples' = [],
            name = 'OLS Default') -> 'OLS information as a list of tuples':
        """Initialize Input Variables"""
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
        self.initialize_content_forms()

    def core_calculations(self):
        """Performs the core calculations for the OLS Regression"""
        #Calculate x_bar, y_bar, beta1, beta1, corr.

    @property
    def x_bar(self):
        return mean(self.x)
    
    @property
    def y_bar(self):
        return mean(self.y)
        
    @property
    def beta1(self):    
        return sum([(i[1] - self.y_bar)*(i[0] - self.x_bar) for i in self.pairs]) / sum([(i[0] - self.x_bar)**2 for i in self.pairs])
    
    @property
    def beta0(self):
        return self.y_bar - self.beta1*self.x_bar
    
    """Calculate y_hate, error, error_squared"""
    @property
    def y_hat(self):    
        return [round(self.beta0 + self.beta1*i, 4) for i in self.x]
    
    @property
    def error(self):
        return  [round(self.y[i] - self.y_hat[i], 4) for i in range(len(self.y))]
    
    @property
    def error_squared(self):
        self.error_squared = [round(i**2, 5) for i in self.error]
    
    def initialize_content_forms(self):
        """Initialize the main groupings of the data: List of Tuples, list of OLS Points"""
        #Create list of tuples containing (x, y, dates, y_hat, error, error_squared)
        self.contents = [list(i) for i in zip(self.x,
                                                self.y,
                                                self.dates,
                                                self.y_hat,
                                                self.error,
                                                self.error_squared)
                                                ]
        #Initialize each tuple as an OLS Point
        self.olspoints = [OLSPoint(i) for i in self.contents]

    def summary(self):
        """Print Statement that outputs Beta, Corr., and n"""
        message = "{}: Beta = {:.2f}, Corr. = {:.2f}, n = {:0d}".format(self.name,
                                                                    self.beta1,
                                                                    self.corr,
                                                                    len(self.x)
                                                                    )
        print(message)

    def ols(self):
        """Return a list of tuples containing (x, y, y_hat, error, error_squared)"""
        return list(zip(self.x, self.y, self.dates, self.y_hat, self.error, self.error_squared))


