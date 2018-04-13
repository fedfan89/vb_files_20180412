from time_decorator import my_time_decorator
import random
import itertools

@my_time_decorator
def my_generator(iterations = 20):
    for i in range(iterations):
        yield i

@my_time_decorator
def my_loop(generator):
    for i in itertools.islice(generator,5):
        print(i)

@my_time_decorator
def random_num_generator(iterations=10**1000):
    for i in range(iterations):
        yield random.normalvariate(0,1)

paul_generator = my_generator()
print(paul_generator, type(paul_generator))
my_loop(paul_generator)
my_loop(paul_generator)
my_loop(paul_generator)
my_loop(paul_generator)
my_loop(paul_generator)
my_loop(paul_generator)

random_nums = random_num_generator()
print(random_nums, type(random_nums))
my_loop(random_nums)
