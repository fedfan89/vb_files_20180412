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

#import price table from pickle file for s&p500 + discretionary symbols
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
def ols_params(pairs: 'list of (x,y) pairs as tuples' = [], name = 'default') -> 'ols information as a list of tuples':
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
            "beta = ", round(beta1, 2), ", ",
            "corr = ", round(corr, 2), ", ",
            "n = ", len(x), ", ",
            sep="")

    #return a list of tuples containing (x, y, y_hat, error, error_squared)
    return list(zip(x, y, y_hat, error, error_squared))

#calculate hv metrics for a list of symbols
@my_time_decorator
def run(symbols: 'list of strings' = ['pg']) -> 'none':
    for sym in symbols:
        #establish lists of relevant numbers
        prices = price_table[sym]
        cc_moves = percent_change(prices)
        cc_moves_initial_scrub = initial_scrub(cc_moves, .05)
        cc_moves_scrubbed = paulization(cc_moves, .05, 3)

        #print standard dev calcs
        print(sym, ": hv calcs \n------------------\n", sep="", end="")
        [print("hv calc: ", round(np.std(i)*math.sqrt(252), 2), ", bizdays: ", len(i), sep="", end="\n")
            for i in [cc_moves, cc_moves_initial_scrub, cc_moves_scrubbed]]
        print("----------------------------\n")

#run(['amzn', 'aapl', 'goog', 'fb'])

class OLSPair(object):
    
    def __init__(self
            , index: 'str'
            , stock: 'str'
            , index_cutoff = .01
            , stock_cutoff = .1
            , pct_cutoff = .15
            , data_points = 500
            , base100 = True
            , beta_setting: 'str'= 'scrubbed'
            , manual_beta: 'float' = 1.0
            , scatter_type='all'
            , bucket_size: 'int' = 10) -> 'list of tuples':

        #import price table from pickle file -> s&p500 + discretionary symbols
        self.price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).head(2000)

        #instantiate passed variables
        self.index = index
        self.stock = stock
        self.index_cutoff = index_cutoff
        self.stock_cutoff = stock_cutoff
        self.pct_cutoff = pct_cutoff
        self.data_points = data_points
        self.base100 = base100
        self.beta_setting = beta_setting
        self.manual_beta = manual_beta
        self.scatter_type = scatter_type
        self.bucket_size = bucket_size

        #print statement declaring the ols regression
        print("ols regression: "
            , self.index, " v. "
            , self.stock, "\n"
            , "---------------------------------------------------------"
            , sep="")

        #attribute: dates in the sample (-> pandas series)
        self.dates = price_table.head(self.data_points).index.values
        self.start_date = self.price_table.head(1).index.values
        self.end_date = self.price_table.tail(1).index.values
        
        #attribute: closing prices for index and stock (-> pandas series)
        self.index_prices = price_table[self.index].head(self.data_points)
        self.stock_prices = price_table[self.stock].head(self.data_points)
        self.dates = price_table.head(self.data_points).index.values
    
        #attribute: cc_moves for index and stock (-> lists of floats)
        self.index_cc_moves = percent_change(self.index_prices)
        self.stock_cc_moves = percent_change(self.stock_prices)

        #attribute: triplets of index prices, stock prices, and dates (-> list of tuples)
        self.raw_prices = list(zip(self.index_prices, self.stock_prices, self.dates))
        
        #attribute: ols pairs of cc_moves (-> list of tuples)
        self.raw_pairs = list(zip(self.index_cc_moves, self.stock_cc_moves))
        self.raw_pairs = ols_params(pairs = self.raw_pairs, name = 'raw pairs')

        #attribute: ols pairs scrubbed based on the abs(index move) > index_cutoff
        self.initial_scrub = [item for item in self.raw_pairs if abs(item[0]) > self.index_cutoff]
        self.inital_scrub = ols_params(pairs = self.initial_scrub, name = 'initial scrub')
        #make an attribute for data points that were scrubbed?
        #make an attribute for count?

        #attribute: ols pairs scrubbed again based on abs(stock moves) < stock_cutoff
        self.second_scrub = [item for item in self.initial_scrub if abs(item[1]) < self.stock_cutoff]
        self.second_scrub = ols_params(pairs = self.second_scrub, name = 'second scrub')
        
        
        #attribute: ols pairs scrubbed again by dropping the x% of data points with the highest error_squared term (takes pct_cutoff parameter)
        n = math.ceil(self.pct_cutoff*len(self.second_scrub))
        error_squared = [i[4] for i in self.second_scrub]
        rank = ss.rankdata(error_squared, method='ordinal')
        self.third_scrub = [self.second_scrub[i] for i in range(len(self.second_scrub)) if rank[i] < (len(self.second_scrub)+1 - n)]
        self.third_scrub = ols_params(pairs = self.third_scrub, name = 'third scrub')
        #scrubbed = [second_scrub[i] for i in range(len(second_scrub)) if rank[i] >= (len(second_scrub)+1 - n)]
    
        #Methodology for Adjusted Stock Price
        #Beta Choice: manual, raw, or scrubbed
        if self.beta_setting == 'manual':
            beta_graph = self.manual_beta
        elif self.beta_setting == 'raw':
            beta_graph = ols_beta(self.raw_pairs)
        elif self.beta_setting == 'scrubbed':
            beta_graph = ols_beta(self.third_scrub)
        else:
            raise ValueError
        
        #create a deep_copy of raw_prices and reverse them
        self.raw_prices_reversed = copy.deepcopy(self.raw_prices)
        self.raw_prices_reversed.reverse()
        
        #Dates (used later in the graph)
        dates = [i[2] for i in self.raw_prices_reversed]
        
        #Prices: base100
        self.raw_index_prices_base100 = [100]
        self.raw_stock_prices_base100 = [100]
        self.adj_stock_prices_base100 = [100]
        
        #Prices: baseStock
        self.raw_index_prices_baseStock = [self.raw_prices_reversed[0][1]]
        self.raw_stock_prices = [i[1] for i in self.raw_prices_reversed]
        self.adj_stock_prices = [self.raw_prices_reversed[0][1]]
        
        #Pct_Moves
        self.raw_index_pct_moves = []
        self.raw_stock_pct_moves = []
        self.adj_stock_pct_moves = []

        for i in range(1,len(self.raw_prices_reversed)):
            #Pct_Moves
            raw_index_pct_move = round((self.raw_prices_reversed[i][0] / self.raw_prices_reversed[i-1][0]) - 1, 3)
            raw_stock_pct_move = round((self.raw_prices_reversed[i][1] / self.raw_prices_reversed[i-1][1]) - 1, 3)
            adj_stock_pct_move = round((1 + raw_stock_pct_move) / (1 + beta_graph*raw_index_pct_move) - 1, 3)
            
            #Append Pct_Moves to Lists
            self.raw_index_pct_moves.append(raw_index_pct_move)
            self.raw_stock_pct_moves.append(raw_stock_pct_move)
            self.adj_stock_pct_moves.append(adj_stock_pct_move)
            
            #Calculate Stock Prices and Append to Lists
            #Prices: base100
            self.raw_index_prices_base100.append(self.raw_index_prices_base100[i-1]*(1+raw_index_pct_move))
            self.raw_stock_prices_base100.append(self.raw_stock_prices_base100[i-1]*(1+raw_stock_pct_move))
            self.adj_stock_prices_base100.append(self.adj_stock_prices_base100[i-1]*(1 + adj_stock_pct_move))
            #Prices: baseStock
            self.raw_index_prices_baseStock.append(self.raw_index_prices_baseStock[i-1]*(1+raw_index_pct_move))
            self.adj_stock_prices.append(self.adj_stock_prices[i-1]*(1 + adj_stock_pct_move))
        
        self.paul_pair = list(zip(self.raw_index_pct_moves, self.raw_stock_pct_moves))
        self.paul_pair = ols_params(self.paul_pair)

#        #Scrub adj_stock_pct_moves using paulization function (used in HV calculations)
#        adj_stock_pct_moves_scrubbed = paulization(adj_stock_pct_moves, .1, 2.75)
#        raw_stock_pct_moves_scrubbed = paulization(raw_stock_pct_moves, .1, 2.75)
#        
#        #Create pairings of the adjusted stock moves and the index
#        adj_pairs = list(zip(raw_index_pct_moves, adj_stock_pct_moves))
#
#        #Absolute Value of the Pct Moves
#        abs_adj_pairs = list(zip([abs(i[0]) for i in adj_pairs], [abs(i[1]) for i in adj_pairs]))
#        
#        #Scrub Outliers
#        abs_adj_pairs_scrubbed = [abs_adj_pairs[i] for i in range(len(abs_adj_pairs)) if abs_adj_pairs[i][0]>.0025 and abs_adj_pairs[i][1] < .05]
#        adj_pairs_scrubbed = [adj_pairs[i] for i in range(len(abs_adj_pairs)) if abs_adj_pairs[i][0]>.0025 and abs_adj_pairs[i][1] < .05]
#    
        
    def scatter_plot(self, pairs=None):
        #Create Scatter Plot of Daily Returns (stock, index)    
        if pairs is None:
            pairs = self.paul_pair
        plt.title('Scatter Plot: ' + str(self.stock) + " v. " + str(self.index))
        scatter(self.third_scrub, label='Third Scrub', color='gold')
        plt.xlabel(str(self.stock))
        plt.ylabel(str(self.index))
        plt.style.use('ggplot')
        plt.legend()
        plt.show()
    
    def daily_returns_scatter_plot(self):
        #Create Scatter Plot of Daily Returns (stock, index)    
        plt.title('Scatter Plot: ' + str(self.index) + " v. " + str(self.stock))
        if self.scatter_type == 'all':
            scatter(self.raw_pairs, label ='Raw Pairs', color='red')
            scatter(self.initial_scrub, label='Initial Scrub', color='blue')
        scatter(self.second_scrub, label='Second Scrub', color='green')
        scatter(self.third_scrub, label='Third Scrub', color='gold')
        plt.xlabel(str(self.index))
        plt.ylabel(str(self.stock))
        plt.style.use('ggplot')
        plt.legend()
        plt.show()
#    
#    def print_hv_calculations():
#        #Calculate and Print HV Calculations
#        print("-------------HV Calculations----------------")
#        HV_raw_index = round(np.std(raw_index_pct_moves)*math.sqrt(252), 2) 
#        HV_raw_stock = round(np.std(raw_stock_pct_moves)*math.sqrt(252), 2)
#        HV_adj_stock = round(np.std(adj_stock_pct_moves)*math.sqrt(252), 2)
#        HV_total = round(math.sqrt((HV_raw_index*beta_graph)**2 + HV_adj_stock**2), 2)
#        HV_adj_stock_scrubbed = round(np.std(adj_stock_pct_moves_scrubbed)*math.sqrt(252), 2)
#        HV_total_fwd = round(math.sqrt((HV_raw_index*beta_graph)**2 + HV_adj_stock_scrubbed**2), 2)
#        print("Raw: ", HV_raw_stock, "\n",
#                "Index: ", HV_raw_index,
#                ", Idio.: ", HV_adj_stock,
#                " <==> Total: ", HV_total, "\n"
#                "Index: ", HV_raw_index,
#                ", Idio. Scrubbed: ", HV_adj_stock_scrubbed,
#                " <==> Fwd Est.: ", HV_total_fwd,
#                sep="")
#    
#    def stock_chart():
#        #Create Stock Chart for stock based on beta to index -> Line Graphs
#        if base100 == True:
#            plt.plot(dates, raw_index_prices_base100, color='black', label=(str(sym1) + " (base100)"))
#            plt.plot(dates, raw_stock_prices_base100, color='red', label = (str(sym2) + " (base100)"))
#            plt.plot(dates, adj_stock_prices_base100, color='gold', label= (str(sym2) + " beta-adjusted (base100)"))
#        else:
#            plt.plot(dates, raw_stock_prices, color='red', label=str(sym2)+ " Raw Prices")
#            plt.plot(dates, raw_index_prices_baseStock, color='black', label=str(sym1) + " (base: " + str(sym2) + ")")
#            plt.plot(dates, adj_stock_prices, color='gold', label=str(sym2) + " Adj. Prices based on Beta to " + str(sym1) + "")
#        #plt.plot(dates, diffs, color = 'black')
#        plt.title(str(sym2) + " Stock Chart" + "; Index: " + str(sym1) + "; Beta Est. = " +str(beta_graph))
#        plt.xlabel("Date")
#        plt.ylabel("Stock Price")
#        plt.xticks(rotation=45)
#        plt.style.use('ggplot')
#        plt.legend()
#        plt.show()
#    
#    #Print Statements
#    def ols_print_statement(pairs: 'list of tuples'=[]) -> 'empty':
#        print("Beta: ", ols_beta(pairs),
#            ", Corr: ", round(np.corrcoef([i[0] for i in pairs], [i[1] for i in pairs])[1,0],2),
#            ", n = ", len(pairs),
#            sep="")
#    
#    def adj_stock_returns_scatter_plot():
#        #Create Scatter Plot of Adjusted Stock Returns v. Index (index, adj_stock)
#        plt.title('Scatter Plot; Adj. Stock Moves v Index: ' + str(sym1) + " v. " + str(sym2))
#        scatter(adj_pairs, "Adj. Stock Returns v. Raw Index", "silver")
#        scatter(abs_adj_pairs, "2", "gold")
#        scatter(abs_adj_pairs_scrubbed, "3", "black")
#        plt.xlabel(str(sym1))
#        plt.ylabel(str(sym2))
#        plt.style.use('ggplot')
#        plt.legend()
#        plt.show()
#
#    
#    #Recommended Series: raw_stock_pct_moves, adj_stock_pct_moves, raw_stock_pct_moves_scrubbed, adj_stock_pct_moves_scrubbed
#    #Inputs
#    def vol_of_vol(series: 'list of pct_moves' = self.raw_stock_pct_moves
#            #, scrubbed = False
#            #, bucket_size = bucket_size
#            #, name = ""
#            ):
#        
#        my_series = series
#        series_size = len(my_series)
#        num_buckets = math.floor(series_size / bucket_size)
#        HV_list = []
#        HV_dates =[]
#
#        if scrubbed == True:
#            #scrub moves over specified_threshold
#            for i in range(len(my_series)):
#                if abs(my_series[i]) > .20:
#                    my_series[i] = math.nan
#            #scrub based on std_dev
#            std_cutoff = np.nanstd(my_series)*2.0
#            print("std_cutoff: ", std_cutoff)
#            for i in range(len(my_series)):
#                if abs(my_series[i]) > std_cutoff:
#                    my_series[i] = math.nan
#
#        for i in range(num_buckets):
#            bucket = my_series[i*bucket_size:(i+1)*bucket_size]
#            HV_calc = round(np.nanstd(bucket)*math.sqrt(252), 2)
#            HV_list.append(HV_calc)
#            date = dates[(i+1)*bucket_size]
#            HV_dates.append(date)
#        vol_of_vol = round(np.nanstd(HV_list)*math.sqrt(252),2)
#        
#        print(name, " <==> ",
#                "Count: ", len(HV_list), "\n",
#                "Vol of Vol: ", vol_of_vol,
#                "HV List: ", HV_list,
#                #"Dates: ", HV_dates
#                sep="")
#        return HV_list
#    
#    def get_HV_dates(series = self.raw_stock_pct_moves, bucket_size = self.bucket_size):
#        HV_dates = []
#        series_size = len(series)
#        num_buckets = math.floor(series_size / bucket_size)
#        for i in range(num_buckets):
#            date = dates[(i+1)*bucket_size]
#            HV_dates.append(date)
#        return HV_dates
#    
#    def throwaway(self):
#        HV_raw = vol_of_vol(raw_stock_pct_moves, False, bucket_size, "Raw")
#        HV_adj = vol_of_vol(adj_stock_pct_moves, False, bucket_size, "Adj.")
#        HV_raw_scrubbed = vol_of_vol(raw_stock_pct_moves, True, bucket_size, "Raw Scrubbed")
#        HV_adj_scrubbed = vol_of_vol(adj_stock_pct_moves, True, bucket_size, "Adj. Scrubbed")
#        HV_index_raw = vol_of_vol(raw_index_pct_moves, False, bucket_size, "Raw")
#        HV_index_scrubbed = vol_of_vol(raw_index_pct_moves, True, bucket_size, "Adj.")
#        
#        HV_dates = get_HV_dates(raw_stock_pct_moves, bucket_size)
#
#        plt.plot(HV_dates, HV_index_raw, color = 'black', label=str(sym1)+ ": Index Raw")
#        #plt.plot(HV_dates, HV_raw, color='black', label=(str(sym2) + ": HV Raw"))
#        #plt.plot(HV_dates, HV_adj, color='red', label = (str(sym2) + ": HV Adj."))
#        #plt.plot(HV_dates, HV_raw_scrubbed, color='gold', label= (str(sym2) + ": Stock Raw Scrubbed"))
#        plt.plot(HV_dates, HV_adj_scrubbed, color='blue', label=str(sym2)+ ": Stock Adj. Scrubbed")
#        plt.title(str(sym2) + " " + str(bucket_size) + " Day Trailing HVs" + "; Index: " + str(sym1) + "; Beta Est. = " +str(beta_graph))
#        plt.xlabel("Date")
#        plt.ylabel(str(bucket_size) + "Day HVs")
#        plt.xticks(rotation=45)
#        plt.style.use('ggplot')
#        plt.legend()
#        plt.show()
#
#        exam_pair = list(zip(HV_index_raw, HV_adj_scrubbed))
#        ols_print_statement(exam_pair)
#
#        #Create Scatter Plot of Adjusted Stock Returns v. Index (index, adj_stock)
#        plt.title('Trailing HVs Scatter Plot: Adj. Stock Moves v Index: ' + str(sym1) + " v. " + str(sym2))
#        plt.scatter(HV_index_raw, HV_adj_scrubbed, color="silver", label="Adj. Stock Returns v. Raw Index")
#        plt.xlabel(str(sym1))
#        plt.ylabel(str(sym2))
#        plt.style.use('ggplot')
#        plt.legend()
#        #plt.show()
#
#    #Commands for Print Statements and Matplotlib Charts
#    #print_hv_calculations()
#    #daily_returns_scatter_plot()
#    #stock_chart()
#
#    #Adjusted_Pct_Moves v Index_Pct_Moves    
#    #ols_print_statement(adj_pairs)
#    #ols_print_statement(abs_adj_pairs)
#    #ols_print_statement(abs_adj_pairs_scrubbed)
#    #adj_stock_returns_scatter_plot()
#
#Indices: SPY, IWM, QQQ, RUT, IBB, XBI, XLP, XRT
#Stocks: AMZN, AAPL, GOOG, FB, MSFT, PG, PFE, XOM, CVX, WMT, ALNY, SRPT, EXEL, MRNS, CRBP, NBIX, BMRN 
#Hash Dictionary for index_cutoff parameter
index_cutoff_hash = {'SPY': .01, 'XLP': .01, 'XBI': .02, 'IBB': .01}
sym1, sym2 = 'SPY', 'WMT'

pair = OLSPair(sym1, sym2,
        index_cutoff = index_cutoff_hash[sym1],
        stock_cutoff=.075,
        pct_cutoff = .1,
        data_points = 50,
        base100=False,
        beta_setting = 'scrubbed',
        manual_beta = 1.0,
        scatter_type = 'all',
        bucket_size = 10
        )
print(pair.third_scrub, "\n"*5, pair.second_scrub)
pair.scatter_plot()

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
