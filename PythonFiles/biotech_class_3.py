import datetime as dt
import pandas as pd

""""
Changes since v2 (biotech_class_2.py):
    -Save instances of Evt_PresElection into a dictionary
    -Before I had assigned each instance a variable name
"""

class SystematicEvent(object):
    name = 'Default'
    abbrev_name = 'Default'
    timing = None
    mult = 1.0
    instances = []
    
    def __init__(self, stock: 'str', move_input: 'float', idio_mult = 1.0):
        self.stock = stock
        self.move_input = move_input
        self.idio_mult = idio_mult
        if type(self).__name__ == 'SystematicEvent':
            self.instances.append(self)
        else:
            self.instances.append(self)
            SystematicEvent.instances.append(self)
        print("{} Instantiated Successfully".format(self.stock))

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
    abbrev_name = 'Election'
    timing = dt.datetime(2020, 11, 3)
    mult = 1.0
    instances = []
    
    def __init__(self, stock: 'str', move_input: 'float', idio_mult = 1.0):
        super().__init__(stock, move_input, idio_mult)

class TakeoutEvent(object):
    name = 'Takeout'
    abbrev_name = 'T.O.'
    timing = None
    mult = 1.0
    instances = []

    def __init__(self, stock: 'str', takeout_rank: 'int'):
        self.stock = stock
        self.takeout_rank = takeout_rank





#-------------------PresElection Calculations-----------------#
# Import PresElection Parameters from CSV File
#PresElectionParams = pd.DataFrame([('SRPT', .05), ('BMRN', .03), ('GILD', .02)], columns = ['Stock', 'move_input'])
PresElectionParams = pd.read_csv("/home/paul/Environments/finance_env/PresElectionParams.csv")
PresElectionParams.set_index('Stock', inplace=True)

# Create dictionary of PresElection Events where the key is the stock symbol
PresElection_Evts = {}
for stock, move_input in PresElectionParams.itertuples():
    PresElection_Evts[stock] = SysEvt_PresElection(stock, move_input)

print(PresElectionParams, PresElection_Evts, SysEvt_PresElection.instances, sep="\n")
PresElection_Evts['SRPT'].set_move_input(.1)
print(PresElection_Evts['SRPT'].__dict__)

