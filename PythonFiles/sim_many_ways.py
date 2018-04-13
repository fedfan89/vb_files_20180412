"""
formula for black scholes distribution:
S_t = S_o*exp(r*t - vol^2/2 + z*vol*sqrt(t)
"""

import math
import random
import numpy as np
from scipy.stats import norm
from time_decorator import my_time_decorator

#function for a single iteration of the black scholes simulation with default z = 0
def list_average(list_of_nums):
    return (sum(list_of_nums)/float(len(list_of_nums)))

def stock_future_func(stock=100, r=0, t=1.0, vol=.40, z=0):
    stock_future = stock*math.exp(r*t - (vol**2)*(t/2) + z*vol*math.sqrt(t))
    return stock_future

#loop through a list and append to create list of random numbers
@my_time_decorator
def rand_number_gen(iterations=10**5):
    rand_numbers=[]
    for i in range(iterations):
        rand = random.normalvariate(0,1)
        rand_numbers.append(rand)
    return rand_numbers

#list comprehension to create a list of random numbers
@my_time_decorator
def rand_number_gen_2(iterations=10**5):
    return [random.normalvariate(0,1) for i in range(iterations)]

#create list of 1s and adjust cells by reference to create list of random numbers
@my_time_decorator
def rand_number_gen_3(iterations=10**5):
    rand_numbers = [1] * iterations
    for i in range(iterations):
        rand_numbers[i] = random.normalvariate(0,1)
    return rand_numbers

#create a generator of random numbers
@my_time_decorator
def rand_number_gen_4(iterations=10**5):
    for i in range(iterations):
        yield random.normalvariate(0,1)

#create a numpy array of random numbers
@my_time_decorator
def rand_number_gen_nparray(iterations=10**5):
    rand_numbers = np.array([random.normalvariate(0,1) for i in range(iterations)])
    return rand_numbers

#loop through a list and append to create list of simulated future stock prices
@my_time_decorator
def bs_simulation(iterations=10**5):
    stock, r, t, vol = 100, 0, 1.0, .40
    stock_futures = []
    for i in range(iterations):
        rand = random.normalvariate(0, 1)
        stock_future = round(stock*math.exp(r*t - (vol**2)*(t/2) + rand*vol*math.sqrt(t)), 2)
        stock_futures.append(stock_future)
    return stock_futures

#loop through a list and append but use a pre-made list of random numbers
@my_time_decorator
def bs_simulation_2(list_of_rand_nums):
    stock, r, t, vol = 100, 0, 1.0, .40
    stock_futures = []
    for rand in list_of_rand_nums:
        stock_future = round(stock*math.exp(r*t - (vol**2)*(t/2) + rand*vol*math.sqrt(t)), 2)
        stock_futures.append(stock_future)
    return stock_futures

#loop through a list and append using a pre-made generator object of random numbers
@my_time_decorator
def bs_simulation_3(generator_object_of_rand_nums):
    stock, r, t, vol = 100, 0, 1.0, .40
    stock_futures = []
    for rand in generator_object_of_rand_nums:
        stock_future = round(stock*math.exp(r*t - (vol**2)*(t/2) + rand*vol*math.sqrt(t)), 2)
        stock_futures.append(stock_future)
    return stock_futures

#create a numpy array of random numbers and maniupate to create a numpy array of simulated future stock prices
@my_time_decorator
def bs_simulation_nparray(iterations=10**5):
    stock, r, t, vol = 100, 0, 1.0, .40
    nparray_of_rand_nums = np.array([random.normalvariate(0,1) for i in range(iterations)])
    stock_futures = 100*np.e**(0*1.0 - (.4**2)*(1.0/2) + nparray_of_rand_nums*.4*np.sqrt(1.0))
    return stock_futures

rand_numbers = rand_number_gen()
rand_numbers_2 = rand_number_gen_2()
rand_numbers_3 = rand_number_gen_3()
rand_numbers_4 = rand_number_gen_4()
rand_numbers_nparray = rand_number_gen_nparray()
print(rand_numbers_nparray)
print(type(rand_numbers_nparray))
stock_futures = bs_simulation()
stock_futures_2 = bs_simulation_2(rand_numbers)
stock_futures_3 = bs_simulation_3(rand_numbers_4)
stock_futures_nparray = bs_simulation_nparray()
print([list_average(l) for l in [rand_numbers, rand_numbers_2, rand_numbers_3, stock_futures, stock_futures_2, stock_futures_3, stock_futures_nparray]])

#sample test code using -1, 0, and 1 standard deviation moves
#print(norm.ppf(.5), norm.ppf(.84), norm.ppf(.16))
#for i in [-1,0,1]:
#    print(stock_future_func(stock=100, r=0, t=1.0, vol=.40, z=i))
