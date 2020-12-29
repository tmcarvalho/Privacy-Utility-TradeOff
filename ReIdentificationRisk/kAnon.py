
def kAnon(obj, keyVars):
    """
    Measure the individual risk base on the frequency of the sample.
    :param obj: input data.
    :param keyVars: names (or indices) of key variables.
    :return: a list with frequency in the sample
    """
    calcFreq(obj=obj, keyVars=keyVars).verify_erros()


class calcFreq():
    def __init__(self, obj, keyVars):
        self.obj = obj
        self.keyVars = keyVars

    def verify_erros(self):
        return self.kAnonWork()

    def kAnonWork(self):
        return self.obj.groupby(self.keyVars, dropna=False).transform('size')