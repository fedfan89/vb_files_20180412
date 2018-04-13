from beta_class_3 import ScrubParams, Beta
from decorators import my_time_decorator
from ols2 import OLS as MainOLS

@my_time_decorator
def run():
    #params = ScrubParams(stock_cutoff = .15, index_cutoff = .01, percentile_cutoff = .8)

    #beta = Beta(stock = 'AAPL', index = 'SPY', lookback = 20, ScrubParams = params)

    #beta1 = Beta('AAPL', 'SPY', 2000, ScrubParams())
    #beta2 = Beta('AAPL', 'SPY', 2000, ScrubParams(.05))
    stock = 'SRPT'
    index = 'XBI'
    count = 2000
    stock_cutoff = .1
    index_cutoff = ''
    percentile_cutoff = .9
    beta0 = Beta(stock, index, count, ScrubParams())
    beta1 = Beta(stock, index, count, ScrubParams(stock_cutoff))
    beta2 = Beta(stock, index, count, ScrubParams(stock_cutoff, .01))
    beta3 = Beta(stock, index, count, ScrubParams(stock_cutoff, .02, percentile_cutoff))
    beta4 = Beta(stock, index, count, ScrubParams(stock_cutoff, .03, percentile_cutoff))
    beta5 = Beta(stock, index, count, ScrubParams(stock_cutoff, .04, percentile_cutoff))
    beta6 = Beta(stock, index, count, ScrubParams(stock_cutoff, .05, percentile_cutoff))
    
    beta0.describe()
    beta1.describe()
    beta2.describe()
    beta3.describe()
    beta4.describe()
    beta5.describe()
    beta6.describe()
    #print(beta4.initial_scrub, beta4.second_scrub, beta4.third_scrub, sep ="\n")
    #print(beta4.OLS_object.contents)
    #print(beta4.main.round(3))
    #print(beta1, beta2, beta3, beta4)
run()
