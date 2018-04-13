import pickle
import copy
import pprint
import decimal
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from time_decorator import my_time_decorator
from ols import OLS
import math
import matplotlib.pyplot as plt
import scipy.stats as ss

#If I structured each Stock Data Point as a Dictionary, what would it look like?
#Keys....
#{'date': , 'index_price': , 'stock_price': , 'index_pct_move': , 'stock_pct_move': , '


#import price table from pickle file for s&p500 + discretionary symbols
price_table = pickle.load(open('sp500_price_table.pkl', 'rb')).head(2000)

#Functions
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

#Calculate HV Metrics for a List of Symbols
@my_time_decorator
def HVs_Multiple_Syms(symbols: 'list of strings' = ['pg']) -> 'none':
    for sym in symbols:
        #establish lists of relevant numbers
        prices = price_table[sym]
        cc_moves = percent_change(prices)
        cc_moves_initial_scrub = initial_scrub(cc_moves, .05)
        cc_moves_scrubbed = paulization(cc_moves, .1, 2.75)

        #print standard dev calcs
        print(sym, ": HV Calcs \n------------------\n", sep="", end="")
        [print("HV: ", round(np.std(i)*math.sqrt(252), 2), ", BizDays: ", len(i), sep="", end="\n")
            for i in [cc_moves, cc_moves_initial_scrub, cc_moves_scrubbed]]
        print("----------------------------\n")

HVs_Multiple_Syms(['AMZN', 'AAPL', 'GOOG', 'FB'])








class Stock(object):
    def __init__(self
            #takes index and stock, scrubbing parameters, and lookback parameter
            , index: 'str'
            , stock: 'str'
            , index_cutoff = .01
            , stock_cutoff = .1
            , pct_cutoff = .15
            , data_points = 500
            
            #create Adjusted Stock Chart
            , base100 = True
            , beta_setting: 'str'= 'scrubbed'
            , manual_beta: 'float' = 1.0
            
            #create Scatter Plot of OLS pairs -> should be its own Class
            , scatter_type= 'all'
            
            #create Rolling HV Calculations
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
        print("OLS Regression: "
            , self.index, " v. "
            , self.stock, "\n"
            , "---------------------------------------------------------"
            , sep="")

        #Attribute: dates in the sample (-> pandas series)
        self.dates = price_table.head(self.data_points).index.values
        self.start_date = self.price_table.head(1).index.values
        self.end_date = self.price_table.tail(1).index.values
        
        #Attribute: closing prices for index and stock (-> pandas series)
        self.index_prices = price_table[self.index].head(self.data_points)
        self.stock_prices = price_table[self.stock].head(self.data_points)
    
        #Attribute: cc_moves for index and stock (-> lists of floats)
        self.index_cc_moves = percent_change(self.index_prices)
        self.stock_cc_moves = percent_change(self.stock_prices)

        #Attribute: triplets of index prices, stock prices, and dates (-> list of tuples)
        self.raw_prices = list(zip(self.index_prices, self.stock_prices, self.dates))
        
        #Attribute: Ols Pairs of cc_moves (and dates) (-> list of tuples)
        self.raw_pairs = OLS(pairs = list(zip(self.index_cc_moves, self.stock_cc_moves, self.dates)), name = 'Raw Pairs')
        
        #Attribute: OLS Object scrubbed based on the abs(index move) > index_cutoff
        self.initial_scrub = OLS([item.contents for item in self.raw_pairs.olspoints if abs(item.x) > self.index_cutoff])
        #make an attribute for scrubbed data points? for count?

        #Attribute: OLS Object scrubbed again based on abs(stock move) < stock_cutoff
        self.second_scrub = OLS([item.contents for item in self.initial_scrub.olspoints if abs(item.y) < self.stock_cutoff])
    
        #Attribute: OLS Object scrubbed again by dropping the x% of data points with the highest error_squared term
        #(Takes pct_cutoff parameter) #I think this could be made more succinct by using sort functionality within the list
        n = math.ceil(self.pct_cutoff*self.second_scrub.count)
        rank = ss.rankdata(self.second_scrub.error_squared, method='ordinal')
        self.third_scrub = OLS([self.second_scrub.contents[i] for i in range(self.second_scrub.count) if rank[i] < (self.second_scrub.count+1 - n)])
        print("SUMMARY HERE", self.third_scrub.contents,"END SUMMARY")
    
    def Scrub_ScatterPlot(self,
                base100 = False,
                beta_setting ='scrubbed',
                manual_beta = 1.0,
                scatter_type = 'all'):
        
        self.base100 = base100
        self.beta_setting = beta_setting
        self.manual_beta = manual_beta
        self.scatter_type = 'all'

        plt.title('Scatter Plot: ' + str(self.index) + " v. " + str(self.stock))
        if self.scatter_type == 'all':
            plt.scatter(self.raw_pairs.x, self.raw_pairs.y, label ='Raw Pairs', color='red')
            plt.scatter(self.initial_scrub.x, self.initial_scrub.y, label='Initial Scrub', color='blue')
        plt.scatter(self.second_scrub.x, self.second_scrub.y, label='Second Scrub', color='green')
        plt.scatter(self.third_scrub.x, self.third_scrub.y, label='Third Scrub', color='gold')
        plt.xlabel(str(self.index))
        plt.ylabel(str(self.stock))
        plt.style.use('ggplot')
        plt.legend()
        plt.show()

    def StockChart(self,
                    base100 = True,
                    beta_setting: 'str' = 'scrubbed',
                    manual_beta: 'float' = 1.0
                    ):

        #Methodology for calculating Adjusted Stock Prices
        #Beta Choice: manual, raw, or scrubbed
        if self.beta_setting == 'manual':
            beta_graph = self.manual_beta
        elif self.beta_setting == 'raw':
            beta_graph = self.raw_pairs.beta1
        elif self.beta_setting == 'scrubbed':
            beta_graph = self.third_scrub.beta1
        else:
            raise ValueError
        
        #Create a deep_copy of raw_prices and reverse the order to make the oldest date the first entry
        self.raw_prices_reversed = copy.deepcopy(self.raw_prices)
        self.raw_prices_reversed.reverse()
        
        #Dates (used later in the graph)
        self.dates_reversed = [i[2] for i in self.raw_prices_reversed]
        
        #Pct_Moves
        self.raw_index_pct_moves = []
        self.raw_stock_pct_moves = []
        self.adj_stock_pct_moves = []
        
        #Prices: base100
        self.raw_index_prices_base100 = [100]
        self.raw_stock_prices_base100 = [100]
        self.adj_stock_prices_base100 = [100]
        
        #Prices: baseStock
        self.raw_index_prices_baseStock = [self.raw_prices_reversed[0][1]]
        self.raw_stock_prices = [i[1] for i in self.raw_prices_reversed]
        self.adj_stock_prices = [self.raw_prices_reversed[0][1]]
        
        for i in range(1,len(self.raw_prices_reversed)):
            #Calculate Pct_Moves
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
        
        #Create Adjusted Stock Chart based on beta to index -> Line Graphs
        if base100 == True:
            plt.plot(self.dates_reversed, self.raw_index_prices_base100, color='black', label=(str(self.index) + " (base100)"))
            plt.plot(self.dates_reversed, self.raw_stock_prices_base100, color='red', label = (str(self.stock) + " (base100)"))
            plt.plot(self.dates_reversed, self.adj_stock_prices_base100, color='gold', label= (str(self.stock) + " beta-adjusted (base100)"))
        else:
            plt.plot(self.dates_reversed, self.raw_stock_prices, color='red', label=str(self.stock)+ " Raw Prices")
            plt.plot(self.dates_reversed, self.raw_index_prices_baseStock, color='black', label=str(self.index) + " (base: " + str(self.index) + ")")
            plt.plot(self.dates_reversed, self.adj_stock_prices, color='gold', label=str(self.stock) + " Adj. Prices based on Beta to " + str(self.stock) + "")
        plt.title(str(self.stock) + " Stock Chart" + "; Index: " + str(self.index) + "; Beta Est. = " +str(round(beta_graph,2)))
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.xticks(rotation=45)
        plt.style.use('ggplot')
        plt.legend()
        plt.show()

        #METHODOLOGY FOR LATER -- LOOKS AT ADJUSTED MOVES V INDEX MOVES
        #self.paul_pair = list(zip(self.raw_index_pct_moves, self.raw_stock_pct_moves))
        #self.paul_pair = OLS(self.paul_pair)
        
        #Total HV Calculations: Scrub adj_stock_pct_moves using paulization function (used in HV calculations)
        #self.adj_stock_pct_moves_scrubbed = paulization(self.adj_stock_pct_moves, .1, 2.75)
        #self.raw_stock_pct_moves_scrubbed = paulization(self.raw_stock_pct_moves, .1, 2.75)

    def RollingHVs(self, bucket: 'int' = 10):
        #Rolling HV Calculations: Evaluate the Rolling HVs of raw_index_pct_moves v. adj_stock_pct_moves
        #1) Create Pairings
        self.adj_pairs = [list(i) for i in zip(self.raw_index_pct_moves, self.adj_stock_pct_moves, self.dates_reversed)]
        #2) Scrub Pairings: set values to NAN if adj_stock_move > specified threshold
        self.adj_pairs_scrubbed = copy.deepcopy(self.adj_pairs)
        for i in range(len(self.adj_pairs)):
            if self.adj_pairs[i][1] > .1:
                self.adj_pairs_scrubbed[i][0] = math.nan
                self.adj_pairs_scrubbed[i][1] = math.nan
        std_cutoff = np.nanstd([i[1] for i in self.adj_pairs_scrubbed])*.025
        for i in range(len(self.adj_pairs_scrubbed)):
            if abs(self.adj_pairs_scrubbed[i][1]) > std_cutoff:
                self.adj_pairs_scrubbed[i][0] = math.nan
                self.adj_pairs_scrubbed[i][1] = math.nan
        print("HELLOOOOOOOOOOOOOOOOOOOOOOOOOO", self.adj_pairs_scrubbed)
        
        
        
        #adj_pairs_scrubbed = [item for item in adj_pairs if adj_pairs[1] < .15]

#        #Absolute Value of the Pct Moves
#        abs_adj_pairs = list(zip([abs(i[0]) for i in adj_pairs], [abs(i[1]) for i in adj_pairs]))
        
        #Scrub Outliers
#        abs_adj_pairs_scrubbed = [abs_adj_pairs[i] for i in range(len(abs_adj_pairs)) if abs_adj_pairs[i][0]>.0025 and abs_adj_pairs[i][1] < .05]
#        adj_pairs_scrubbed = [adj_pairs[i] for i in range(len(abs_adj_pairs)) if abs_adj_pairs[i][0]>.0025 and abs_adj_pairs[i][1] < .05]
    
    ###I need to work on this if I want to use it.    
    def Adj_ScatterPlot(self, pairs=None):
        #Create Scatter Plot of Daily Returns (stock, index)    
        if pairs is None:
            pairs = self.adj_pairs_scrubbed
        plt.title('Scatter Plot: ' + str(self.stock) + " v. " + str(self.index))
        scatter(pairs, label='Third Scrub', color='gold')
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
    #Recommended Series: raw_stock_pct_moves, adj_stock_pct_moves, raw_stock_pct_moves_scrubbed, adj_stock_pct_moves_scrubbed
    #Inputs
    def HV_List(self, pct_moves: 'list of pct_moves' = None
            , scrubbed = False
            , initial_cutoff = .10
            , std_cutoff = 2.75
            , bucket_size = None
            , name = ""
            ):
        
        if pct_moves is None:
            pct_moves = self.raw_stock_pct_moves
        if bucket_size is None:
            bucket_size = self.bucket_size

        series_size = len(pct_moves)
        num_buckets = math.floor(series_size / bucket_size)
        HV_list = []
        ##HV_dates =[]

        if scrubbed == True:
            #scrub moves over specified_threshold
            for i in range(len(pct_moves)):
                if abs(pct_moves[i]) > initial_cutoff:
                    pct_moves[i] = math.nan
            #scrub based on std_dev
            std_pct_cutoff = np.nanstd(my_series)*std_cutoff
            for i in range(len(pct_moves)):
                if abs(pct_moves[i]) > std_pct_cutoff:
                    pct_moves[i] = math.nan

        for i in range(num_buckets):
            bucket = pct_moves[i*bucket_size:(i+1)*bucket_size]
            HV_calc = round(np.nanstd(bucket)*math.sqrt(252), 2)
            HV_list.append(HV_calc)
            ##date = self.dates_reversed[(i+1)*bucket_size]
            ##HV_dates.append(date)
        vol_of_vol = round(np.nanstd(HV_list)*math.sqrt(252),2)
        
        print(name, " <==> ",
                "Count: ", len(HV_list), "\n",
                "Vol of Vol: ", vol_of_vol,
                "HV List: ", HV_list,
                #"Dates: ", HV_dates
                sep="")
        return HV_list
    
    def get_HV_dates(self, series = None, bucket_size = None):
        if series is None:
            series = self.raw_stock_pct_moves
        if bucket_size is None:
            bucket_size = self.bucket_size
        
        HV_dates = []
        series_size = len(series)
        num_buckets = math.floor(series_size / bucket_size)
        for i in range(num_buckets):
            date = self.dates_reversed[(i+1)*bucket_size]
            HV_dates.append(date)
        return HV_dates
    
#    HV_raw = HV_List(self.raw_stock_pct_moves, False, self.bucket_size, "Raw")
#    HV_adj = HV_List(self.adj_stock_pct_moves, False, self.bucket_size, "Adj.")
#    HV_raw_scrubbed = HV_list(self.raw_stock_pct_moves, True, self.bucket_size, "Raw Scrubbed")
#    HV_adj_scrubbed = HV_List(self.adj_stock_pct_moves, True, self.bucket_size, "Adj. Scrubbed")
#    HV_index_raw = HV_List(self.raw_index_pct_moves, False, self.bucket_size, "Raw")
#    HV_index_scrubbed = HV_List(self.raw_index_pct_moves, True, self.bucket_size, "Adj.")
#    
#    HV_dates = get_HV_dates(raw_stock_pct_moves, bucket_size)
#
#    plt.plot(HV_dates, HV_index_raw, color = 'black', label=str(sym1)+ ": Index Raw")
#    #plt.plot(HV_dates, HV_raw, color='black', label=(str(sym2) + ": HV Raw"))
#    #plt.plot(HV_dates, HV_adj, color='red', label = (str(sym2) + ": HV Adj."))
#    #plt.plot(HV_dates, HV_raw_scrubbed, color='gold', label= (str(sym2) + ": Stock Raw Scrubbed"))
#    plt.plot(HV_dates, HV_adj_scrubbed, color='blue', label=str(sym2)+ ": Stock Adj. Scrubbed")
#    plt.title(str(sym2) + " " + str(bucket_size) + " Day Trailing HVs" + "; Index: " + str(sym1) + "; Beta Est. = " +str(beta_graph))
#    plt.xlabel("Date")
#    plt.ylabel(str(bucket_size) + "Day HVs")
#    plt.xticks(rotation=45)
#    plt.style.use('ggplot')
#    plt.legend()
#    plt.show()
#
#    exam_pair = list(zip(HV_index_raw, HV_adj_scrubbed))
#    ols_print_statement(exam_pair)
#
#    #Create Scatter Plot of Adjusted Stock Returns v. Index (index, adj_stock)
#    plt.title('Trailing HVs Scatter Plot: Adj. Stock Moves v Index: ' + str(sym1) + " v. " + str(sym2))
#    plt.scatter(HV_index_raw, HV_adj_scrubbed, color="silver", label="Adj. Stock Returns v. Raw Index")
#    plt.xlabel(str(sym1))
#    plt.ylabel(str(sym2))
#    plt.style.use('ggplot')
#    plt.legend()
#    plt.show()
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
sym1, sym2 = 'XBI', 'SRPT'
my_pairing = Stock(sym1,
            sym2,
            index_cutoff = index_cutoff_hash[sym1],
            stock_cutoff=.075,
            pct_cutoff = .1,
            data_points = 50,
            #base100=False,
            #beta_setting = 'scrubbed',
            #manual_beta = 1.0,
            #scatter_type = 'all',
            bucket_size = 10
            )

my_pairing.Scrub_ScatterPlot( scatter_type = 'all')
my_pairing.StockChart(base100 = False, beta_setting = 'scrubbed', manual_beta = 1.0)


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
