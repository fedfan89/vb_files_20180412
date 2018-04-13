#--------------------------
def funct1():
    print("This works.")

def funct2():
    funct1()

funct2()


#--------------------------
class Paul(object):
    def funct3():
        print("This doesn't work")

    def funct4():
        funct3()

Paul.funct4()
