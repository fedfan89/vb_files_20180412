"""
formula for black scholes distribution:
S_t = S_o*exp(r*t - vol^2/2 + z*vol*sqrt(t)"""

import math
import random
import numpy as np
import itertools
import tkinter as tk
import matplotlib.pyplot as plt
from scipy.stats import norm
from time_decorator import my_time_decorator

#compute the average value in a list
def list_average(list_of_nums):
    return (sum(list_of_nums)/float(len(list_of_nums)))

#create a numpy array of random numbers
@my_time_decorator
def create_rand_nums(iterations:'int'=10**5) -> 'np_array':
    return np.array([random.normalvariate(0,1) for i in range(iterations)])

#run simulation using a previously cretaed array of random numbers as the input
@my_time_decorator
def bs_simulation(rand_nums: 'np_array') -> 'np_array':
    stock, r, t, vol = 100, 0, 1.0, .40
    stock_futures = stock*np.e**(r*t - (vol**2)*(t/2) + rand_nums*vol*np.sqrt(t))
    return stock_futures

rand_nums = create_rand_nums(iterations = 10**6)
stock_futures = bs_simulation(rand_nums)
print([list_average(l) for l in [rand_nums, stock_futures]])

############################# graph component ##########################3
stock = 100
bins = [float(i) for i in np.arange(stock*.2, stock*3.0, stock*.05)]
plt.hist(stock_futures, bins, histtype='bar', rwidth=0.8, color='purple')

plt.xlabel('St. Dev. Moves')
plt.ylabel('Relative Frequency')
plt.title('Beautiful Probabilitiy Distribution\nSmooth and Pretty!\n{:,d} Iterations'.format(len(stock_futures)))
plt.legend()
plt.show()
