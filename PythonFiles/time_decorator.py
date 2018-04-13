import time

def my_time_decorator(original_function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = original_function(*args, **kwargs)
        t2 = time.time()
        diff = round(t2 - t1, 3)
        print("time in seconds:", diff)
        return result
    return wrapper


