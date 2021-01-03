import numpy as np


def kAnonCalc(obj, keyVars):
    """
    Measure the individual risk base on the frequency of the sample.
    :param obj: input data.
    :param keyVars: names (or indices) of key variables.
    :return: a list with frequency in the sample
    """
    return calcFreq(obj=obj, keyVars=keyVars).verify_errors()


class calcFreq:
    def __init__(self, obj, keyVars):
        self.obj = obj
        self.keyVars = keyVars

    def verify_errors(self):
        columns = list(self.obj.columns.values)
        error_vars = np.setdiff1d(self.keyVars, columns)
        if len(error_vars) != 0:
            raise ValueError("[" + '%s' % ', '.join(map(str, error_vars)) + "] specified in 'keyVars' can "
                                                                            "not be found!\n")
        else:
            return self.kAnonWork()

    def kAnonWork(self):
        fk = self.obj.groupby(self.keyVars)[self.keyVars[0]].transform(len)
        return fk


def calc_max_risk(obj):
    """
    Calculate the percentage of fk when is 1 in a dataframe
    :param obj: input data
    :return: percentage of fk when is 1
    """
    # exclude the target variable in order to perform the risk calculation
    obj_val = obj[obj.columns[:-1]]
    # list with all columns names
    keyVars = list(obj_val.columns.values)
    # get frequency of keyVars
    fk = kAnonCalc(obj=obj_val, keyVars=keyVars)
    return (fk == 1).sum() * 100 / len(fk)
