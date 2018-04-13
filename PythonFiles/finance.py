import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from bs4 import BeautifulSoup
import sklearn
style.use('ggplot')

"""
--------------------------------------------------------------------
The functions in this module are web scraping functions.
    1) Pull S&P 500 symbols from Wikipedia (and create pickle file).
        --get_sp500_symbols_from_wiki()    

    2) Get historical data for a single stock from Yahoo Finance
        --get_prices(symbol, start, end)

    3) Get historical data for multiple stocks from Yahoo Finance
        --make_price_table(symbols,start,end)

    4) Make pickle file for stock prices for a single symbol
        --pickle_prices(symbol, start, end)
--------------------------------------------------------------------
"""

#pull S&P 500 symbols from Wikipedia
def get_sp500_symbols_from_wiki():
    #pull symbols from Wikipedia table
    sp500_symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')[0][0][1:].reset_index(drop=True)
    discretionary_symbols = pd.Series(['SPY', 'IWM', 'QQQ', 'RUT', 'IBB', 'XBI', 'XLP', 'XRT', 'ALNY', 'SRPT', 'EXEL', 'MRNS', 'CRBP', 'BMRN', 'NBIX'])
    sp500_symbols = pd.concat([sp500_symbols, discretionary_symbols]).sort_values().reset_index(drop=True)

    #save to csv
    sp500_symbols.to_csv('sp500_symbols.csv')

    #save to pickle
    pickle_file = open('sp500_symbols.pkl', 'wb')
    pickle.dump(sp500_symbols, pickle_file, pickle.HIGHEST_PROTOCOL)
    pickle_file.close()

    return sp500_symbols.tolist()
    

#function to get prices from Yahoo's website for one symbol
def get_prices(symbol: 'str'='AAPL', start: 'dt.datetime' = dt.datetime(2016, 1, 1), end: 'dt.datetime' = dt.datetime.today()):
    count = 1
    while count < 10:
        try:
            df = web.DataReader(symbol, 'yahoo', start, end).round(2)
        except Exception:
            count += 1
            if count == 9:
                print(str(symbol), ' failed the query')
                failed_symbols.append(symbol)
            else:
                pass
            continue
        else:
            break
    
    print(symbol, ": ", count, sep="")
    yahoo_query_count.append(count)

    if count < 10:
        prices = pd.DataFrame(df['Adj Close'])
        prices = prices.rename(columns = {'Adj Close' : symbol})
        return(prices)
    else:
        return

#function to get prices from Yahoo's website for multiple symbols
def make_price_table(symbols: 'list', start = dt.datetime(2016,1,1), end = dt.datetime.today(), name = 'default_price_table'):
    yahoo_query_count = []
    failed_symbols = []
    price_table = get_prices(symbols[0], start, end)
    for symbol in symbols[1:]:
        try:
            price_table = price_table.join(get_prices(symbol, start, end))
        except Exception:
            pass
        else:
            pass
    print(yahoo_query_count, failed_symbols, end='\n')
    
    #save price table and query details to csv file
    price_table.to_csv(name)
    yahoo_query_count.to_csv(str(name) + '_query_count.csv')
    failed_symbols.to_csv(str(name) + '_failed_symbols.csv')

    #save price table to pickle file
    pickle_file = open(str(name) + '_price_table.pkl', 'wb')
    pickle.dump(price_table, pickle_file, pickle.HIGHEST_PROTOCOL)
    pickle_file.close()

    return(price_table)

#function to download stock prices to pickle file using Yahoo's website for one symbol
def pickle_prices(symbol, start, end):
    df = web.DataReader(symbol, 'yahoo', start, end).round(2)
    prices = df['Adj Close']
    pickle_file = open(str(symbol) + '_pickle_file.pkl', 'wb')
    pickle.dump(prices, pickle_file, pickle.HIGHEST_PROTOCOL)
    pickle_file.close()

#---------------------------------------------------------------------------------------
#These queries use the functions above.
symbols = get_sp500_symbols_from_wiki()
#symbols = pickle.load(open('sp500_symbols.pkl', 'rb'))
price_table = make_price_table(symbols, start = dt.datetime(2016,1,1), end = dt.datetime.today())
print(price_table)
