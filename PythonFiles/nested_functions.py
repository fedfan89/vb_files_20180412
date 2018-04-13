class Paul(object):
    fail = []
    def funct1():
        fail.append("You failed again.")

    def funct2():
        return funct1

Paul.funct2()
