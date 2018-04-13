import datetime as dt

class Stock(object):
    def __init__(self, stock, **kwargs):
        self.stock = stock
        
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)

#SRPT = Stock('SRPT', sector = 'Healthcare', subsector = 'Biotech')
#print(SRPT, SRPT.sector, SRPT.subsector)

class Event(object):
    def __init__(self, name, timing):
        self.name = name
        self.timing = timing

    def __str__(self):
        return self.name

#election = Event('Presidential_Election', dt.datetime(2018, 5, 1))
#print(election, election.name, election.timing)

class StockEvent(Event):
    def __init__(self, name, timing, stock):
        super().__init__(name, timing)
        self.stock = stock

#election_srpt = StockEvent('Presidential_Election', dt.datetime(2018, 5, 1), 'SRPT')

#print(election_srpt, election_srpt.name, election_srpt.timing, election_srpt.stock)

class SystematicEvent(object):
    name = 'Default'
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
        return self.stock

    @property
    def modeled_move(self):
        return self.mult*self.idio_mult*self.move_input

    def set_idio_mult(self, new_value):
        self.idio_mult = new_value

    def set_move_input(self, new_value):
        self.move_input = new_value


class Evt_PresElection(SystematicEvent):
    name = 'U.S. Presidential Election'
    timing = dt.datetime(2020, 11, 3)
    mult = 1.0
    instances = []
    
    def __init__(self, stock: 'str', move_input: 'float', idio_mult = 1.0):
        super().__init__(stock, move_input, idio_mult)

evt1 = Evt_PresElection('SRPT', .05)
evt2 = Evt_PresElection('BMRN', .02)
evt3 = SystematicEvent('GILD', .05)

print(evt1.name, evt1.stock, evt1.move_input)
evt1.set_idio_mult(2.0)
evt1.set_move_input(.1)
print(Evt_PresElection.instances)
print(SystematicEvent.instances)
