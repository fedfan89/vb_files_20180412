import pickle
import pprint
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from time_decorator import my_time_decorator
import math
import matplotlib.pyplot as plt

#suggested tickers: AMZN, AAPL, GOOG, FB, MSFT, PG, PFE, XOM, CVX, WMT, SPY, XBI, IBB
price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).tail(2000)
#pprint.pprint(price_table)

#@my_time_decorator
def percent_change(numbers: 'list of floats') -> 'list of floats':
    return [round((numbers[i] - numbers[i-1])/numbers[i], 3) for i in range(1, len(numbers))]

#@my_time_decorator
def initial_scrub(numbers: 'list of floats', cutoff: 'float' = .15) -> 'list of floats':
    return [num for num in numbers if abs(num)<= cutoff]

#@my_time_decorator
def paulization(numbers: 'list of floats', cutoff: 'float' = .15, std_cutoff: 'float' = 2.0) -> 'list':
    initial = [number for number in numbers if abs(number) <= cutoff]
    std_perc_cutoff = np.std(initial)*std_cutoff
    return [num for num in initial if abs(num) <= std_perc_cutoff]

@my_time_decorator
def run(symbols: 'list of strings' = ['PG']) -> 'None':
    for sym in symbols:
        #establish lists of relevant numbers
        prices = price_table[sym]
        cc_moves = percent_change(prices)
        cc_moves_initial_scrub = initial_scrub(cc_moves, .05)
        cc_moves_scrubbed = paulization(cc_moves, .05, 3)

        #print standard dev calcs
        print(sym, ": HV Calcs \n------------------\n", sep="", end="")
        [print("HV Calc: ", round(np.std(i)*math.sqrt(252), 2), ", BizDays: ", len(i), sep="", end="\n")
            for i in [cc_moves, cc_moves_initial_scrub, cc_moves_scrubbed]]
        print("----------------------------\n")

#run(['AMZN', 'AAPL', 'GOOG', 'FB'])

def corr_pair(sym1: 'str', sym2: 'str', index_cutoff = .01, stock_cutoff = .1) -> 'list of tuples':
    #create arrays of prices for the index and stock (-> pandas Series)
    prices1, prices2 = price_table[sym1], price_table[sym2]
    
    #create arrays of cc_moves (-> lists of floats)
    cc_moves1, cc_moves2 = percent_change(prices1), percent_change(prices2)
    
    #create pairs of cc_moves (-> list of tuples)
    raw_pairs = list(zip(cc_moves1, cc_moves2))

    #scrub the list based on the abs(index move) > .01
    initial_scrub = [item for item in raw_pairs if abs(item[0]) > index_cutoff]
    
    #scrub the list again based on the abs(stock moves) < .10
    second_scrub = [item for item in initial_scrub if abs(item[1]) < stock_cutoff]

    #raw cc_move pairs in individual list form
    a, b = [item[0] for item in raw_pairs], [item[1] for item in raw_pairs]
    
    #second_scrub cc_move pairs in individual list form
    x, y = [item[0] for item in second_scrub], [item[1] for item in second_scrub]

    print("orig. corr.: ", round(np.corrcoef(a,b)[1,0], 3), ", scrubbed corr: ", round(np.corrcoef(x,y)[1,0], 3), sep="")
    
    #run regression to determine beta
    #xx, yy = pd.DataFrame(x), pd.DataFrame(y)
    #result = sm.OLS(yy, xx).fit()
    #print(result.summary())
    
    #create scatter plot
    plt.scatter(x, y, label='stock moves scatter plot', color='k')
    plt.xlabel(str(sym1))
    plt.ylabel(str(sym2))
    plt.title('cc_moves pair')
    plt.legend()
    #plt.show()

    #calculate original beta manually (raw pairs)
    a_bar, b_bar = (sum(a)/len(a)), (sum(b)/len(b))
    beta1 = sum([(i[1] - b_bar)*(i[0] - a_bar) for i in raw_pairs]) / sum([(i[0] - a_bar)**2 for i in raw_pairs])
    beta0 = b_bar - beta1*a_bar
    corr = np.corrcoef(a, b)[1,0]
    print("raw pairs--- ", "beta1:", round(beta1, 3),", beta0:", round(beta0, 5), ", count: ", len(a), ", correlation: ", round(corr,3), sep="")
    
    #calculate correlation (corr), beta (beta1), alpha (beta0) manually
    x_bar, y_bar = (sum(x)/len(x)), (sum(y)/len(y))
    beta1 = sum([(i[1] - y_bar)*(i[0] - x_bar) for i in second_scrub]) / sum([(i[0] - x_bar)**2 for i in second_scrub])
    beta0 = y_bar - beta1*x_bar
    corr = np.corrcoef(x, y)[1,0]
    print("second scrub---- ", "beta1:", round(beta1, 3),", beta0:", round(beta0, 5), ", count: ", len(x), ", correlation: ", round(corr, 3),  sep="")
    
    #calculate y_hat, error terms
    y_hat = [beta0 + beta1*i[0] for i in second_scrub]
    error = [i[1] - (beta0 + beta1*i[0]) for i in second_scrub]
    error_squared = [i**2 for i in error]
    
    #create a tuple with x, y, y_hat, error, error squared, correlation
    second_scrub = list(zip(x, y, y_hat, error, error_squared))

    #drop pairs with the highest error term
    max_error = max(error_squared)
    scrubbed_point = [i for i in second_scrub if i[4] == max_error]
    #print(max_error, scrubbed_point)
    third_scrub = [i for i in second_scrub if i[4] < max_error]
    x3, y3 = [item[0] for item in third_scrub], [item[1] for item in third_scrub]
    #print(scrubbed_point)
    
    #create scatter plot
    plt.scatter(x3, y3, label='stock moves scatter plot without biggest error term', color='k')
    plt.xlabel(str(sym1))
    plt.ylabel(str(sym2))
    plt.title('cc_moves pair')
    plt.legend()
    #plt.show()
    
    #calculate beta manually
    x3_bar, y3_bar = (sum(x3)/len(x3)), (sum(y3)/len(y3))
    beta1 = sum([(i[1] - y3_bar)*(i[0] - x3_bar) for i in third_scrub]) / sum([(i[0] - x3_bar)**2 for i in third_scrub])
    beta0 = y3_bar - beta1*x3_bar
    corr = np.corrcoef(x3, y3)[1,0]
    print("after deleting biggest outlier---- ", "beta1:", round(beta1, 3),", beta0:", round(beta0, 5), ", count: ", len(x3), ", correlation: ", round(corr, 3), sep="")

#corr_pair('SPY', 'SRPT', index_cutoff=.0125, stock_cutoff=.075)
corr_pair('IBB', 'SRPT', index_cutoff = .012525, stock_cutoff=.075)
#corr_pair('SPY', 'IBB', index_cutoff = .0125, stock_cutoff=.1)

#in SPY, calculate the number of stocks that were up on the day
#pprint.pprint(price_table['SPY'])
def run2():
    symbols = price_table.columns.values.tolist()
    dates = price_table['SPY'].index.values.tolist()
    price_table_shifted = price_table.copy(deep=True)
    for symbol in symbols:
        for i in range(len(dates[:-1])):
            price_table_shifted[symbol][dates[i]] = (price_table[symbol][dates[i]] - price_table[symbol][dates[i+1]]) / price_table[symbol][dates[i]]
    price_table_shifted = price_table_shifted.round(3)
    price_table_shifted = price_table_shifted.drop(dates[-1])

    pprint.pprint(price_table_shifted)
    up_syms_by_date = []
    for date in dates[:-1]:
        count = 0
        index = 'SPY'
        for symbol in symbols:
            if price_table_shifted[symbol][date] > 0:
                count +=1
        index_move = price_table_shifted[index]
        pair = (date, count, price_table_shifted[index][date])
        up_syms_by_date.append(pair)
        print(date, ": ", count, ", ", index, " move: ", price_table_shifted[index][date], sep="")

#create scatter plot
    index_moves = [i[2] for i in up_syms_by_date]
    count = [i[1] for i in up_syms_by_date]
    print(index_moves)
    print(count)
    plt.scatter(index_moves, count, label = "my scat", color = 'k')
    plt.xlabel('pct move in index')
    plt.ylabel('count')
    plt.title('my scat')
    plt.legend()
    plt.show()

#run2()
