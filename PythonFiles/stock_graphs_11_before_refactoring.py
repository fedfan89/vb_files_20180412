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
"""Params Layout
Each stock is an object -- highest level
-Stock
-Data Calculation Date Range
--StockGraph([list of StockLines'],
                name: 'str'
                )
    --StockLine(Pairing: 'tuple'
                Beta: 'OLSBeta or float',
                Base: 'index, stock, or float',
                DateRange: 'int',
                To_Graph: 'stock or index'
                )
        --Pairing(Stock': 'str',
                Index: 'str'
                )
        --Beta
            --OLSBeta(Pairing: 'tuple',
                    DateRange: 'int',
                    ScrubParams: 'obj'
                    )
                --ScrubObject
                    --None(empty)
                    --InitialScrub(index_cutoff: 'float')
                    --SecondScrub(index_cutoff: 'float',
                                    stock_cutoff: 'float'
                                    )
                    --ThirdScrub(index_cutoff: 'float',
                                    stock_cutoff: 'float',
                                    'pct_cutoff': 'float'
                                    )
--ScatterPlot([list of OLS objects]
    --OLS([list of (x, y, date) triplets],
                    name: 'str'
                    )
        --(x, y, date) triplet


-For a Stock Graph, there are multiple ways to shock 
"""

#think this structure through more
class Params(object):
    def __init__(self, CoreParams, StockChartParams, ScatterPlotParams):
        self.CoreParams = CoreParams
        self.StockChartParams = StockChartParams
        self.ScatterPlotParams = ScatterPlotParams

class CoreParams(object):
    def __init__(self, index, stock, index_cutoff, stock_cutoff, pct_cutoff, data_points):
        self.index = index
        self.stock = stock
        self.index_cutoff = index_cutoff
        self.stock_cutoff = stock_cutoff
        self.pct_cutoff = pct_cutoff
        self.data_points = data_points
    

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
        self.initial_scrub = OLS([item.contents for item in self.raw_pairs.olspoints if abs(item.x) > self.index_cutoff], name='Initial Scrub')
        #make an attribute for scrubbed data points? for count?

        #Attribute: OLS Object scrubbed again based on abs(stock move) < stock_cutoff
        self.second_scrub = OLS([item.contents for item in self.initial_scrub.olspoints if abs(item.y) < self.stock_cutoff], name='Second Scrub')
    
        #Attribute: OLS Object scrubbed again by dropping the x% of data points with the highest error_squared term
        #(Takes pct_cutoff parameter) #I think this could be made more succinct by using sort functionality within the list
        n = math.ceil(self.pct_cutoff*self.second_scrub.count)
        rank = ss.rankdata(self.second_scrub.error_squared, method='ordinal')
        self.third_scrub = OLS([self.second_scrub.contents[i] for i in range(self.second_scrub.count) if rank[i] < (self.second_scrub.count+1 - n)], name='Third Scrub')
    
    def Scrub_ScatterPlot(self,
                base100 = False,
                beta_setting ='scrubbed',
                manual_beta = 1.0,
                scatter_type = 'all'):
        self.base100 = base100
        self.beta_setting = beta_setting
        self.manual_beta = manual_beta
        self.scatter_type = 'all'
        
        self.raw_pairs.summary()
        self.initial_scrub.summary()
        self.second_scrub.summary()
        self.third_scrub.summary()

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
            self.beta_graph = self.manual_beta
        elif self.beta_setting == 'raw':
            self.beta_graph = self.raw_pairs.beta1
        elif self.beta_setting == 'scrubbed':
            self.beta_graph = self.third_scrub.beta1
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
            adj_stock_pct_move = round((1 + raw_stock_pct_move) / (1 + self.beta_graph*raw_index_pct_move) - 1, 3)
            
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
        plt.title(str(self.stock) + " Stock Chart" + "; Index: " + str(self.index) + "; Beta Est. = " +str(round(self.beta_graph,2)))
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.xticks(rotation=45)
        plt.style.use('ggplot')
        plt.legend()
        plt.show()

    def NecessaryFunctionality(self):
        #Rolling HV Calculations: Evaluate the Rolling HVs of Index Moves v. Adjusted Stock Moves
        #1) Create Pairings
        self.adj_pairs = [list(i) for i in zip(self.raw_index_pct_moves, self.adj_stock_pct_moves, self.dates_reversed)]
        
        #2) Scrub Pairings: Set values to NAN if adj_stock_move > specified threshold
        self.adj_pairs_scrubbed = copy.deepcopy(self.adj_pairs)
        for i in range(len(self.adj_pairs)):
            if self.adj_pairs[i][1] > .1:
                self.adj_pairs_scrubbed[i][0] = math.nan
                self.adj_pairs_scrubbed[i][1] = math.nan
        std_cutoff = np.nanstd([i[1] for i in self.adj_pairs_scrubbed])*2.5
        for i in range(len(self.adj_pairs_scrubbed)):
            if abs(self.adj_pairs_scrubbed[i][1]) > std_cutoff:
                self.adj_pairs_scrubbed[i][0] = math.nan
                self.adj_pairs_scrubbed[i][1] = math.nan

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

    def RollingHVsAsList(self,
                        pct_moves: 'list of pct_moves' = None,
                        scrubbed = False,
                        initial_cutoff = .1,
                        std_cutoff = 2.75,
                        bucket_size = None,
                        name = ""
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
            #Scrub moves over specified_threshold
            for i in range(len(pct_moves)):
                if abs(pct_moves[i]) > initial_cutoff:
                    pct_moves[i] = math.nan
            #Scrub based on std_dev
            std_pct_cutoff = np.nanstd(pct_moves)*std_cutoff
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
                "Vol of Vol: ", vol_of_vol, "\n"
                "HV List: ", HV_list,
                #"Dates: ", HV_dates
                sep="")
        return HV_list

    def get_HVs_dates(self, series = None, bucket_size = None):
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
    
    def RollingHVsLineGraph(self, bucket_size: 'int' = None): 
        if bucket_size is None:
            bucket_size = self.bucket_size

        HVs_raw = self.RollingHVsAsList(self.raw_stock_pct_moves, False, .10, 2.75, bucket_size, "Raw")
        HVs_adj = self.RollingHVsAsList(self.adj_stock_pct_moves, False, .10, 2.75, bucket_size, "Adj.")
        HVs_raw_scrubbed = self.RollingHVsAsList(self.raw_stock_pct_moves, True, .10, 2.75, bucket_size, "Raw Scrubbed")
        HVs_adj_scrubbed = self.RollingHVsAsList(self.adj_stock_pct_moves, True, .10, 2.75, bucket_size, "Adj. Scrubbed")
        HVs_index_raw = self.RollingHVsAsList(self.raw_index_pct_moves, False, .10, 2.75, bucket_size, "Raw")
        HVs_index_scrubbed = self.RollingHVsAsList(self.raw_index_pct_moves, True, .10, 2.75, bucket_size, "Adj.")
        
        HVs_dates = self.get_HVs_dates(self.raw_stock_pct_moves, bucket_size)

        plt.plot(HVs_dates, HVs_index_raw, color = 'red', label=str(self.index)+ ": Index Raw")
        plt.plot(HVs_dates, HVs_raw, color='black', label=(str(self.stock) + ": Stock Raw"))
        plt.plot(HVs_dates, HVs_adj, color='green', label = (str(self.stock) + ": Stock Adj."))
        plt.plot(HVs_dates, HVs_raw_scrubbed, color='blue', label= (str(self.stock) + ": Stock Raw Scrubbed"))
        plt.plot(HVs_dates, HVs_adj_scrubbed, color='gold', label=str(self.stock)+ ": Stock Adj. Scrubbed")
        plt.title(str(self.stock) + " " + str(bucket_size) + "  Day Trailing HVs" + "; Index: " + str(sym1) + "; Beta Est. = " +str(round(self.beta_graph, 2)))
        plt.xlabel("Date")
        plt.ylabel(str(bucket_size) + "Day HVs")
        plt.xticks(rotation=45)
        plt.style.use('ggplot')
        plt.legend()
        plt.show()

    #The Calculations for this function are broken
    def Print_HV_Calculations(self):
        #Total HV Calculations: Scrub adj_stock_pct_moves using paulization function (used in HV calculations)
        self.adj_stock_pct_moves_scrubbed = paulization(self.adj_stock_pct_moves, .1, 2.75)
        self.raw_stock_pct_moves_scrubbed = paulization(self.raw_stock_pct_moves, .1, 2.75)
        
        #Calculate and Print HV Calculations
        print("-------------HV Calculations----------------")
        self.HV_raw_index = round(np.std(self.raw_index_pct_moves)*math.sqrt(252), 2) 
        self.HV_raw_stock = round(np.std(self.raw_stock_pct_moves)*math.sqrt(252), 2)
        self.HV_adj_stock = round(np.std(self.adj_stock_pct_moves)*math.sqrt(252), 2)
        self.HV_total = round(math.sqrt((self.HV_raw_index*self.beta_graph)**2 + self.HV_adj_stock**2), 2)
        self.HV_adj_stock_scrubbed = round(np.std(self.adj_stock_pct_moves_scrubbed)*math.sqrt(252), 2)
        self.HV_total_fwd = round(math.sqrt((self.HV_raw_index*self.beta_graph)**2 + self.HV_adj_stock_scrubbed**2), 2)
        print("Raw: ", self.HV_raw_stock, "\n",
                "Index: ", self.HV_raw_index,
                ", Idio.: ", self.HV_adj_stock,
                " <==> Total: ", self.HV_total, "\n"
                "Index: ", self.HV_raw_index,
                ", Idio. Scrubbed: ", self.HV_adj_stock_scrubbed,
                " <==> Fwd Est.: ", self.HV_total_fwd,
                sep="")
    
#----------------------------------------------------------------------------------------------
#POTENTIALLY LOOK AT THIS CODE LATER 
#        #Absolute Value of the Pct Moves
#        abs_adj_pairs = list(zip([abs(i[0]) for i in adj_pairs], [abs(i[1]) for i in adj_pairs]))
        #Scrub Outliers
#        abs_adj_pairs_scrubbed = [abs_adj_pairs[i] for i in range(len(abs_adj_pairs)) if abs_adj_pairs[i][0]>.0025 and abs_adj_pairs[i][1] < .05]
#        adj_pairs_scrubbed = [adj_pairs[i] for i in range(len(abs_adj_pairs)) if abs_adj_pairs[i][0]>.0025 and abs_adj_pairs[i][1] < .05]
    
        #METHODOLOGY FOR LATER -- LOOKS AT ADJUSTED MOVES V INDEX MOVES
        #self.paul_pair = list(zip(self.raw_index_pct_moves, self.raw_stock_pct_moves))
        #self.paul_pair = OLS(self.paul_pair)
#----------------------------------------------------------------------------------------------    


#Indices: SPY, IWM, QQQ, RUT, IBB, XBI, XLP, XRT
#Stocks: AMZN, AAPL, GOOG, FB, MSFT, PG, PFE, XOM, CVX, WMT, ALNY, SRPT, EXEL, MRNS, CRBP, NBIX, BMRN 
#Hash Dictionary for index_cutoff parameter
index_cutoff_hash = {'SPY': .01, 'IWM': .01, 'RUT': .01, 'XLP': .01, 'XBI': .02, 'IBB': .01, 'AAL': .01, 'DAL': .01}
sym1, sym2 = 'SPY', 'PFE'
sym3, sym4 = 'XBI', 'EXEL'
my_pairing = Stock(sym1,
            sym2,
            index_cutoff = index_cutoff_hash[sym1],
            stock_cutoff=.075,
            pct_cutoff = .1,
            data_points = 2000,
            #base100=False,
            #beta_setting = 'scrubbed',
            #manual_beta = 1.0,
            #scatter_type = 'all',
            bucket_size = 15
            )

my_pairing_2 = Stock(sym3,
            sym4,
            index_cutoff = index_cutoff_hash[sym1],
            stock_cutoff=.075,
            pct_cutoff = .1,
            data_points = 5000,
            #base100=False,
            #beta_setting = 'scrubbed',
            #manual_beta = 1.0,
            #scatter_type = 'all',
            bucket_size = 15
            )

#my_pairing.Scrub_ScatterPlot(scatter_type = 'all')
my_pairing.StockChart(base100 = False, beta_setting = 'scrubbed', manual_beta = 1.0)
my_pairing.raw_pairs.summary()
my_pairing.initial_scrub.summary()
my_pairing.second_scrub.summary()
my_pairing.third_scrub.summary()
print(my_pairing.third_scrub.olspoints[0])
#my_pairing.NecessaryFunctionality()
#my_pairing.Adj_ScatterPlot()
#my_pairing.RollingHVsLineGraph()
my_pairing.Print_HV_Calculations()
#my_pairing_2.Scrub_ScatterPlot(scatter_type = 'all')
#my_pairing_2.StockChart(base100 = False, beta_setting = 'scrubbed', manual_beta = 1.0)

print(help(Stock))
print(Stock.__dict__)
print(my_pairing.__dict__)











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
