from decorators import initializer

class Employee(object):
    
    @initializer   
    def __init__(self, first, last, pay):
        pass

paul = Employee('Paul', 'Wainer', 12000)
print(paul.first, paul.last, paul.pay)
