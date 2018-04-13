from functools import wraps
import inspect
import time

#for initializer
from functools import wraps
import inspect

def my_time_decorator(original_function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = original_function(*args, **kwargs)
        t2 = time.time()
        diff = round(t2 - t1, 3)
        print("time in seconds:", diff)
        return result
    return wrapper

#Decorator that automatically assigns the parameters in the init function
#def initializer(func):

    
def initializer(func):
    names, varargs, keywords, defaults = inspect.getargspec(func)
    
    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)
        
        for name, default in zip(reversed(names), reversed(defaults)):
                if not hasattr(self, name):
                    setattr(self, name, default)
        func(self, *args, **kargs)
        
    return wrapper
