from dis import dis #'disassemble' module

class Lyle():
    def __init__(self):
        self.func1()

    def func1(self):
        print("Hello Lyle")

lyle = Lyle()

def paul():
    print("Hello there.")

def billy(x):
    print(x)
    return(x)

dis(billy)

