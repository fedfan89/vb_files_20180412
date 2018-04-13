from beta_class_3 import ScrubParams, Beta
from decorators import my_time_decorator
from ols2 import OLS as MainOLS

@my_time_decorator
def run():
    #params = ScrubParams(stock_cutoff = .15, index_cutoff = .01, percentile_cutoff = .8)

    #beta = Beta(stock = 'AAPL', index = 'SPY', lookback = 20, ScrubParams = params)

    #beta1 = Beta('AAPL', 'SPY', 2000, ScrubParams())
    #beta2 = Beta('AAPL', 'SPY', 2000, ScrubParams(.05))
    stock = 'PG'
    index = 'SPY'
    count = 2000
    stock_cutoff = .075
    #index_cutoff = ''
    percentile_cutoff = .8

    Beta(stock, index, count, ScrubParams(stock_cutoff, .01, percentile_cutoff)).scrub_trajectory()
    
    def betas_by_index_cutoff(index_cutoffs):
        raw = Beta(stock, index, count, ScrubParams())
        raw.describe()

        initial_scrub = Beta(stock, index, count, ScrubParams(stock_cutoff))
        initial_scrub.describe()
        
        for i in index_cutoffs:
            params = ScrubParams(stock_cutoff, i, percentile_cutoff)
            beta = Beta(stock, index, count, params)
            beta.describe()

    index_cutoffs = [.01, .02, .03]
    #betas_by_index_cutoff(index_cutoffs)



    #print(beta4.initial_scrub, beta4.second_scrub, beta4.third_scrub, sep ="\n")
    #print(beta4.OLS_object.contents)
    #print(beta4.main.round(3))
    #print(beta1, beta2, beta3, beta4)
run()
