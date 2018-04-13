import paul_resources
from paul_resources import Aaron, tprint
from biotech_class_4 import Event

class Paul(object):
    """This is the documentation string for Paul."""
    pass

p = Paul()

def main_function():
    pass

evt1 = Event()
evt1.hello = 'HelloThere'
evt1.name='Sup'
print(Event.__dict__)
print(evt1.__dict__)
print(Paul.__class__, p.__class__)

print(dir(Paul), "\n-------------------")
print(dir(p), "\n''''''''''''''''''")

print(vars(Paul) is Paul.__dict__, vars(Paul) == Paul.__dict__)
print(vars(Paul))
print(p.__dict__)
print(p.__doc__)

print(Paul.__dict__, "\n''''''''''''''''''''")
tprint(Paul.__dict__['__dict__'])

print(Paul().__dict__)


print(Paul, Paul()) 
print(paul_resources.Aaron, paul_resources.Aaron())
print(Aaron, Aaron())
print(Paul.__name__, Aaron.__name__, main_function.__name__)
print("HERE:", str(p), repr(p), str(p) is repr(p), str(p) == repr(p))
print(Paul.__module__, Aaron.__module__, main_function.__module__)
print(p.__module__)
print(Paul.__doc__)

print(paul_resources.daily_returns)
print(main_function)

