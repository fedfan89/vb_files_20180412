import pickle
import copy
import pprint
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from time_decorator import my_time_decorator
import math
import matplotlib.pyplot as plt
import scipy.stats as ss

#import price table from pickle file for S&P500 + discretionary symbols
price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).head(2000)

#define functions
def scatter(pairs: 'list of tuples'=[], label = "", color = 'red') -> 'empty':
    plt.scatter([i[0] for i in pairs], [i[1] for i in pairs], label = label, color = color)

def percent_change(numbers: 'list of floats') -> 'list of floats':
    return [round((numbers[i] - numbers[i-1])/numbers[i], 3) for i in range(1, len(numbers))]

def initial_scrub(numbers: 'list of floats', cutoff: 'float' = .15) -> 'list of floats':
    return [num for num in numbers if abs(num)<= cutoff]

def paulization(numbers: 'list of floats', initial_cutoff: 'float' = .15, std_cutoff: 'float' = 2.0) -> 'list':
    initial = [number for number in numbers if abs(number) <= initial_cutoff]
    std_perc_cutoff = np.std(initial)*std_cutoff
    return [num for num in initial if abs(num) <= std_perc_cutoff]

def ols_beta(pairs: 'list of (x,y) pairs as tuples' = []) -> 'float':
    x, y = [i[0] for i in pairs], [i[1] for i in pairs]
    x_bar, y_bar = (sum(x)/len(x)), (sum(y)/len(y))
    return round(sum([(i[1] - y_bar)*(i[0] - x_bar) for i in pairs]) / sum([(i[0] - x_bar)**2 for i in pairs]), 2)

#take a list of (x,y) pairs and return a list of tuples containing (x, y, y_hat, error, error_squared)
#along the way, calculate correlation (corr), beta (beta1), alpha (beta0), error, and error_squared
def ols_params(pairs: 'list of (x,y) pairs as tuples' = [], name = 'default') -> 'OLS information as a list of tuples':
    x, y = [i[0] for i in pairs], [i[1] for i in pairs]
    x_bar, y_bar = (sum(x)/len(x)), (sum(y)/len(y))
    beta1 = sum([(i[1] - y_bar)*(i[0] - x_bar) for i in pairs]) / sum([(i[0] - x_bar)**2 for i in pairs])
    beta0 = y_bar - beta1*x_bar
    corr = np.corrcoef(x, y)[1,0]
    
    #calculate y_hat, error, error_squared
    y_hat = [round(beta0 + beta1*i, 4) for i in x]
    error = [round(y[i] - y_hat[i], 4) for i in range(len(y))]
    error_squared = [round(i**2, 5) for i in error]
    
    print(name, " <=> ",
            "Beta = ", round(beta1, 2), ", ",
            "Corr = ", round(corr, 2), ", ",
            "n = ", len(x), ", ",
            sep="")

    #return a list of tuples containing (x, y, y_hat, error, error_squared)
    return list(zip(x, y, y_hat, error, error_squared))

#calculate HV metrics for a list of symbols
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

def corr_pair(sym1: 'str', sym2: 'str', index_cutoff = .01, stock_cutoff = .1, pct_cutoff = .15, data_points = 500, base100=True, beta_setting: 'str'= 'scrubbed', manual_beta: 'float' = 1.0, scatter_type='all') -> 'list of tuples':
    print("OLS Regression: "
            , sym2, " v. ", sym1, "\n"
            , "---------------------------------------------------------"
            , sep="")
    #create arrays of prices for the index and stock (-> pandas Series)
    prices1, prices2, dates = price_table[sym1].head(data_points), price_table[sym2].head(data_points), price_table.head(data_points).index.values
    
    #create arrays of cc_moves (-> lists of floats)
    cc_moves1, cc_moves2 = percent_change(prices1), percent_change(prices2)

    #create list of tuples of raw prices with the date (-> list of tuples)
    raw_prices = list(zip(prices1, prices2, dates))
    
    #create pairs of cc_moves (-> list of tuples)
    raw_pairs = list(zip(cc_moves1, cc_moves2))
    raw_pairs = ols_params(pairs = raw_pairs, name = 'Raw Pairs')

    #scrub the list based on the abs(index move) > .01
    initial_scrub = [item for item in raw_pairs if abs(item[0]) > index_cutoff]
    inital_scrub = ols_params(pairs = initial_scrub, name = 'Initial Scrub')
   
    #scrub the list again based on the abs(stock moves) < .10
    second_scrub = [item for item in initial_scrub if abs(item[1]) < stock_cutoff]
    second_scrub = ols_params(pairs = second_scrub, name = 'Second Scrub')

    #drop the x% of data points with the highest error_squared term (takes pct_cutoff parameter)
    n = math.ceil(pct_cutoff*len(second_scrub))
    error_squared = [i[4] for i in second_scrub]
    rank = ss.rankdata(error_squared, method='ordinal')
    third_scrub = [second_scrub[i] for i in range(len(second_scrub)) if rank[i] < (len(second_scrub)+1 - n)]
    third_scrub = ols_params(pairs = third_scrub, name = 'Third Scrub')
    scrubbed = [second_scrub[i] for i in range(len(second_scrub)) if rank[i] >= (len(second_scrub)+1 - n)]
    
    
    #Create Scatter Plot of Daily Returns (stock, index)    
    plt.title('Scatter Plot: ' + str(sym1) + " v. " + str(sym2))
    if scatter_type == 'all':
        scatter(raw_pairs, label ='Raw Pairs', color='red')
        scatter(initial_scrub, label='Initial Scrub', color='blue')
    scatter(second_scrub, label='Second Scrub', color='green')
    scatter(third_scrub, label='Third Scrub', color='gold')
    plt.xlabel(str(sym1))
    plt.ylabel(str(sym2))
    plt.style.use('ggplot')
    plt.legend()
    plt.show()

    #Create Stock Chart based on beta to the index
    if beta_setting == 'manual':
        beta_graph = manual_beta
    elif beta_setting == 'raw':
        beta_graph = ols_beta(raw_pairs)
    elif beta_setting == 'scrubbed':
        beta_graph = ols_beta(third_scrub)
    else:
        raise ValueError
    
    raw_prices_reversed = copy.deepcopy(raw_prices)
    raw_prices_reversed.reverse()
    raw_index_prices_base100, raw_stock_prices_base100 = [100], [100]
    adjusted_stock_prices_base100, adjusted_stock_prices = [100], [raw_prices_reversed[0][1]]
    raw_stock_prices = [i[1] for i in raw_prices_reversed]
    raw_index_pct_moves, raw_stock_pct_moves, adj_stock_pct_moves = [], [], []
    raw_index_prices_baseStock = [raw_prices_reversed[0][1]]

    for i in range(1,len(raw_prices_reversed)):
        #pct_moves
        raw_index_pct_move = (raw_prices_reversed[i][0] / raw_prices_reversed[i-1][0]) - 1
        raw_stock_pct_move = (raw_prices_reversed[i][1] / raw_prices_reversed[i-1][1]) - 1
        adj_stock_pct_move = (1 + raw_stock_pct_move) / (1 + beta_graph*raw_index_pct_move) - 1
        
        #append pct_moves to lists
        raw_index_pct_moves.append(raw_index_pct_move)
        raw_stock_pct_moves.append(raw_stock_pct_move)
        adj_stock_pct_moves.append(adj_stock_pct_move)
        
        #calculate prices and appends to lists
        raw_index_prices_base100.append(raw_index_prices_base100[i-1]*(1+raw_index_pct_move))
        raw_stock_prices_base100.append(raw_stock_prices_base100[i-1]*(1+raw_stock_pct_move))
        adjusted_stock_prices_base100.append(adjusted_stock_prices_base100[i-1]*(1 + adj_stock_pct_move))
        adjusted_stock_prices.append(adjusted_stock_prices[i-1]*(1 + adj_stock_pct_move))
        raw_index_prices_baseStock.append(raw_index_prices_baseStock[i-1]*(1+raw_index_pct_move))
    #scrub the adj_stock_pct_moves using the paulization function
    adj_stock_pct_moves_scrubbed = paulization(adj_stock_pct_moves, .20, 2.75)
    dates, raw_dependent, = [i[2] for i in raw_prices_reversed], [i[1] for i in raw_prices_reversed]
    
    #Calculate and Print HV Calculations
    print("-------------HV Calculations----------------")
    HV_raw_index = round(np.std(raw_index_pct_moves)*math.sqrt(252), 2) 
    HV_raw_stock = round(np.std(raw_stock_pct_moves)*math.sqrt(252), 2)
    HV_adj_stock = round(np.std(adj_stock_pct_moves)*math.sqrt(252), 2)
    HV_total = round(math.sqrt((HV_raw_index*beta_graph)**2 + HV_adj_stock**2), 2)
    HV_adj_stock_scrubbed = round(np.std(adj_stock_pct_moves_scrubbed)*math.sqrt(252), 2)
    HV_total_fwd = round(math.sqrt((HV_raw_index*beta_graph)**2 + HV_adj_stock_scrubbed**2), 2)
    print("Raw: ", HV_raw_stock, "\n",
            "Index: ", HV_raw_index,
            ", Idio.: ", HV_adj_stock,
            " <==> Total: ", HV_total, "\n"
            "Index: ", HV_raw_index,
            ", Idio. Scrubbed: ", HV_adj_stock_scrubbed,
            " <==> Fwd Est.: ", HV_total_fwd,
            sep="")
    
    #Create Stock Charts -> Line Graphs
    if base100 == True:
        plt.plot(dates, raw_index_prices_base100, color='black', label=(str(sym1) + " (base100)"))
        plt.plot(dates, raw_stock_prices_base100, color='red', label = (str(sym2) + " (base100)"))
        plt.plot(dates, adjusted_stock_prices_base100, color='gold', label= (str(sym2) + " beta-adjusted (base100)"))
    else:
        plt.plot(dates, raw_stock_prices, color='red', label=str(sym2)+ " Raw Prices")
        plt.plot(dates, raw_index_prices_baseStock, color='black', label=str(sym1) + " (base: " + str(sym2) + ")")
        plt.plot(dates, adjusted_stock_prices, color='gold', label=str(sym2) + " Adj. Prices based on Beta to " + str(sym1) + "")
    #plt.plot(dates, diffs, color = 'black')
    plt.title(str(sym2) + " Stock Chart" + "; Index: " + str(sym1) + "; Beta Est. = " +str(beta_graph))
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.xticks(rotation=45)
    plt.style.use('ggplot')
    plt.legend()
    plt.show()
    


    #Calculate the correlation between the magnitudes of index and idiosyncratic moves
    adj_pairs = list(zip(raw_index_pct_moves, adj_stock_pct_moves))

    #x_values_abs, y_values_abs = [abs(i) for i in x_values], [abs(i) for i in y_values]
    abs_adj_pairs = list(zip([abs(i[0]) for i in adj_pairs], [abs(i[1]) for i in adj_pairs]))
    
    #Scrub Outliers
    abs_adj_pairs_scrubbed = [abs_adj_pairs[i] for i in range(len(abs_adj_pairs)) if abs_adj_pairs[i][0]>.0025 and abs_adj_pairs[i][1] < .05]
    
    #Print Statements
    def ols_print_statement(pairs: 'list of tuples'=[]) -> 'empty':
        print("Beta: ", ols_beta(pairs),
            ", Corr: ", round(np.corrcoef([i[0] for i in pairs], [i[1] for i in pairs])[1,0],2),
            ", n = ", len(pairs),
            sep="")
    ols_print_statement(adj_pairs)
    ols_print_statement(abs_adj_pairs)
    ols_print_statement(abs_adj_pairs_scrubbed)
    
    #Create Scatter Plot of Adjusted Stock Returns v. Index (index, adj_stock)
    plt.title('Scatter Plot; Adj. Stock Moves v Index: ' + str(sym1) + " v. " + str(sym2))
    scatter(adj_pairs, "Adj. Stock Returns v. Raw Index", "silver")
    scatter(abs_adj_pairs, "2", "gold")
    scatter(abs_adj_pairs_scrubbed, "3", "black")
    plt.xlabel(str(sym1))
    plt.ylabel(str(sym2))
    plt.style.use('ggplot')
    plt.legend()
    plt.show()

#Indices: SPY, IWM, QQQ, RUT, IBB, XBI, XLP, XRT
#Stocks: AMZN, AAPL, GOOG, FB, MSFT, PG, PFE, XOM, CVX, WMT, ALNY, SRPT, EXEL, MRNS, CRBP, NBIX, BMRN 
#Hash Dictionary for index_cutoff parameter
index_cutoff_hash = {'SPY': .01, 'XLP': .01, 'XBI': .02, 'IBB': .01}
sym1, sym2 = 'XBI', 'SRPT'
corr_pair(sym1, sym2,
        index_cutoff = index_cutoff_hash[sym1],
        stock_cutoff=.075,
        pct_cutoff = .1,
        data_points = 252,
        base100=True,
        beta_setting = 'scrubbed',
        manual_beta = 1.0,
        scatter_type = 'all'
        )

##------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#In SPY, calculate the number of stocks that were up on the day
#pprint.pprint(price_table['SPY'])
def run_SPY_members():
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

#Create Scatter Plot
    index_moves = [i[2] for i in up_syms_by_date]
    count = [i[1] for i in up_syms_by_date]
    plt.scatter(index_moves, count, label = "my scat", color = 'k')
    plt.xlabel('pct move in index')
    plt.ylabel('count')
    plt.title('my scat')
    plt.legend()
    plt.show()

#run_SPY_members
