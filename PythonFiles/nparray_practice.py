import numpy as np
import random

a = np.zeros(10)
print(a, type(a), len(a), a[0], type(a[0]))

b = (a + 2)*10
print(b,type(b), len(b))

c = a + random.normalvariate(0,1)
print(c, type(c), len(c))

d = np.arange(0,10, 1)
print(d, type(d), len(d))

f = b + d
print(f, type(f), len(f))

g = np.e**f
print(g, type(g), len(g))
