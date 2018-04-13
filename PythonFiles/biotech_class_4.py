import datetime as dt
import pandas as pd
from paul_resources import InformationTable, tprint
""""
Changes since v2 (biotech_class_2.py):
    -Save instances of Evt_PresElection into a dictionary
    -Before I had assigned each instance a variable name
"""

class Event(object):
    name = 'General Event'
    abbrev_name = 'GenEvent'
    timing = None
    instances = ['Event']
    #main_lever = 2.0

    def __init__(self):
        for cls in type(self).__mro__[0:-1]:
            cls.instances.append(self)
        
        if type(self).__name__ == 'Event':
            print("General Event Instantiated Successfully")

    def __str__(self):
        return "{}".format(self.abbrev_name)

    def __repr__(self):
        return "{}".format(self.abbrev_name)


class SystematicEvent(Event):
    name = 'Systematic Event'
    abbrev_name = 'SysEvent'
    timing = None
    mult = 1.0
    instances = ['SystematicEvent']
    
    def __init__(self, stock: 'str', move_input: 'float', idio_mult = 1.0):
        super().__init__()
        self.stock = stock
        self.move_input = move_input
        self.idio_mult = idio_mult
        #print("{} {} Instantiated Successfully".format(self.stock, self.name))
        
        if type(self).__name__ == 'SystematicEvent':
            print("{} Systematic Event Instantiated Successfully".format(self.stock))
        
    def __str__(self):
        return "{} ({:.2f}% move)".format(self.name, self.modeled_move*100)

    def __repr__(self):
        return "{} ({})".format(self.abbrev_name, self.stock)

    @property
    def modeled_move(self):
        return self.mult*self.idio_mult*self.move_input

    def set_idio_mult(self, new_value):
        self.idio_mult = new_value

    def set_move_input(self, new_value):
        self.move_input = new_value


class SysEvt_PresElection(SystematicEvent):
    name = 'U.S. Presidential Election'
    abbrev_name = 'Elec.'
    timing = dt.datetime(2020, 11, 3)
    mult = 1.0
    instances = ['Presidential Election']
    
    def __init__(self, stock: 'str', move_input: 'float', idio_mult = 1.0):
        super().__init__(stock, move_input, idio_mult)
        
        print("{} Presidential Election Event Instantiated Successfully".format(self.stock))

class TakeoutEvent(Event):
    name = 'Takeout'
    abbrev_name = 'T.O.'
    timing = None
    mult = 1.0
    instances = []
    
    takeout_buckets = pd.read_csv('/home/paul/Environments/finance_env/TakeoutBuckets.csv')
    takeout_buckets.set_index('Rank', inplace=True)

    base_takeout_premium = .40
    base_mcap = 10000
    mcap_sensitivity = .35

    def __init__(self, stock: 'str', takeout_bucket: 'int'):
        super().__init__()
        self.stock = stock
        self.takeout_bucket = takeout_bucket
        print("{} Takeout Event Instantiated Successfully.".format(self.stock))

    def __str__(self):
        return "{}-{} ({})".format(self.abbrev_name, self.takeout_bucket, self.stock)
    
    def __repr__(self):
        return "{} ({})".format(self.abbrev_name, self.stock)

    @property
    def takeout_prob(self):
        return self.takeout_buckets.loc[self.takeout_bucket, 'Prob']
    
    @property
    def mcap(self):
        try:
            return InformationTable.loc[self.stock, 'Market Cap']    
        except Exception:
            print("{} did not register a Market Cap. Check error source.".format(self.stock))
            return 10000

    @property
    def takeout_premium_adjustment(self):
        return min(((1 / (self.mcap/self.base_mcap)) - 1)*self.mcap_sensitivity, 1.5)

    @property
    def takeout_premium(self):
        return self.base_takeout_premium * (1 + self.takeout_premium_adjustment)

if __name__ == "__main__":
    #-------------------PresElection Setup-----------------#
    PresElectionParams = pd.read_csv("/home/paul/Environments/finance_env/PresElectionParams.csv")
    PresElectionParams.set_index('Stock', inplace=True)

    # Create PresElection Events Dict
    PresElection_Evts = {}
    for stock, move_input in PresElectionParams.itertuples():
        PresElection_Evts[stock] = SysEvt_PresElection(stock, move_input)

    #-------------------Takeout Setup-----------------#
    TakeoutParams = pd.read_csv("TakeoutParams.csv")
    TakeoutParams.set_index('Stock', inplace=True)

    # Create Takeout Events Dict
    Takeout_Evts = {}
    for stock, bucket in TakeoutParams.itertuples():
        Takeout_Evts[stock] = TakeoutEvent(stock, bucket)

    takeout_dict = {}
    for stock, event in Takeout_Evts.items():
        takeout_dict[stock] = (event.takeout_prob, event.takeout_premium)

    takeout_df = pd.DataFrame(takeout_dict).T.round(3)
    takeout_df.rename(columns = {0: 'Prob', 1: 'Premium'}, inplace=True)
    takeout_df.rename_axis('Stock', inplace=True)
    
    
    evt = SystematicEvent('ZFGN', .20)
    evt2 = Event()
    evt3 = SysEvt_PresElection('GM', .05)
    evt4 = TakeoutEvent('NBIX', 1)
    
    print("\n\n\nAll Events---\n", Event.instances, "\n")
    print("Systematic Event---\n", SystematicEvent.instances, "\n")
    print("Presidential Election---\n", SysEvt_PresElection.instances,"\n")
    print("Takeout Event---\n", TakeoutEvent.instances,"\n")
    print(takeout_df.sort_values('Premium', ascending=False))
