import datetime as dt

class Stock(object):
    def __init__(self, stock, **kwargs):
        self.stock = stock
        
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)

SRPT = Stock('SRPT', sector = 'Healthcare', subsector = 'Biotech')
print(SRPT, SRPT.sector, SRPT.subsector)

class Event(object):
    def __init__(self, name, timing):
        self.name = name
        self.timing = timing

    def __str__(self):
        return self.name

election = Event('Presidential_Election', dt.datetime(2018, 5, 1))
print(election, election.name, election.timing)

class StockEvent(Event):
    def __init__(self, name, timing, stock):
        super().__init__(name, timing)
        self.stock = stock

election_srpt = StockEvent('Presidential_Election', dt.datetime(2018, 5, 1), 'SRPT')

print(election_srpt, election_srpt.name, election_srpt.timing, election_srpt.stock)


class Evt_PresElection(object):
    name = 'U.S. Presidential Election'
    timing = dt.datetime(2020, 11, 3)
    mult = 1.0
    instances = []
    
    def __init__(self, stock: 'str', move_input: 'float', idio_mult = 1.0):
        self.stock = stock
        self.move_input = move_input
        self.idio_mult = idio_mult
        Evt_PresElection.instances.append(self)
        print("Instantiation Successful:", self)

    def __str__(self):
        return "{} ({:.2f}% move)".format(Evt_PresElection.name, self.modeled_move*100)

    def __repr__(self):
        return self.stock

    @property
    def modeled_move(self):
        return Evt_PresElection.mult*self.idio_mult*self.move_input

    def set_idio_mult(self, new_value):
        self.idio_mult = new_value

    def set_move_input(self, new_value):
        self.move_input = new_value

evt1 = Evt_PresElection('SRPT', .05)
print(evt1.name, evt1.stock, evt1.move_input)
print(Evt_PresElection.mult, evt1.mult)
evt1.set_idio_mult(2.0)
evt1.set_move_input(.1)
print(evt1.idio_mult, evt1.move_input, evt1.modeled_move)
print(evt1)
print(Evt_PresElection.name)
print(Evt_PresElection.instances)
print(Evt_PresElection.instances[0].move_input)
evt1.set_move_input(.3)
print(Evt_PresElection.instances[0].move_input)
evt2 = Evt_PresElection('BMRN', .02)
print(Evt_PresElection.instances)
Evt_PresElection.mult = 5.0
print(evt2.modeled_move)

