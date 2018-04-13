"""
formula for black scholes distribution:
S_t = S_o*exp(r*t - vol^2/2 + z*vol*sqrt(t)
"""

import math
import random
import numpy as np
import itertools
from scipy.stats import norm
from time_decorator import my_time_decorator

#function for a single iteration of the black scholes simulation with default z = 0
def list_average(list_of_nums):
    return (sum(list_of_nums)/float(len(list_of_nums)))

def stock_future_func(stock=100, r=0, t=1.0, vol=.40, z=0):
    stock_future = stock*math.exp(r*t - (vol**2)*(t/2) + z*vol*math.sqrt(t))
    return stock_future

#create a (very long) generator of random numbers
@my_time_decorator
def rand_number_gen_generator(iterations=10**50):
    for i in range(iterations):
        yield random.normalvariate(0,1)

#create a numpy array of random numbers
@my_time_decorator
def rand_number_gen_nparray(iterations=10**4):
    rand_numbers = np.array([random.normalvariate(0,1) for i in range(iterations)])
    return rand_numbers

#create a numpy array of random numbers and maniupate to create a numpy array of simulated future stock prices
@my_time_decorator
def bs_simulation_nparray(iterations=10**4):
    stock, r, t, vol = 100, 0, 1.0, .40
    nparray_of_rand_nums = np.array([random.normalvariate(0,1) for i in range(iterations)])
    stock_futures = 100*np.e**(0*1.0 - (.4**2)*(1.0/2) + nparray_of_rand_nums*.4*np.sqrt(1.0))
    return stock_futures

@my_time_decorator
def bs_simulation(list_of_rand_nums):
    stock, r, t, vol = 100, 0, 1.0, .40
    stock_futures = 100*np.e**(0*1.0 - (.4**2)*(1.0/2) + list_of_rand_nums*.4*np.sqrt(1.0))
    return stock_futures

#create an array of future stock prices using a previously made generator of random numbers
@my_time_decorator
def bs_simulation_nparray_2(iterations=10**4):
    stock, r, t, vol = 100, 0, 1.0, .40
    nparray_of_rand_nums = np.array([i for i in itertools.islice(rand_numbers, iterations)])
    stock_futures = 100*np.e**(0*1.0 - (.4**2)*(1.0/2) + nparray_of_rand_nums*.4*np.sqrt(1.0))
    return stock_futures

rand_numbers = rand_number_gen_nparray()
stock_futures_nparray = bs_simulation_nparray()
stock_futures = bs_simulation(rand_numbers)
print([list_average(l) for l in [rand_numbers, stock_futures_nparray, stock_futures]])
