import random
#from random import choices
from random import choices

value = random.random()
print(value)

greetings = ['Hello', 'Hi', 'Hey']

#value = choices(greetings, k=2)
value = choices(greetings)
print(value)
