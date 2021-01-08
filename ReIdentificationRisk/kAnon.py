
def kAnonCalc(obj):
    """
    Measure the individual risk base on the frequency of the sample.
    :param obj: input dataframe.
    :return: a list with frequency in the sample.
    """
    return calcFreq(obj=obj).kAnonWork()


class calcFreq:
    def __init__(self, obj):
        self.obj = obj

    def kAnonWork(self):
        keyVars = list(self.obj.columns.values)
        fk = self.obj.groupby(keyVars)[keyVars[0]].transform(len)
        return fk


def calc_max_risk(obj):
    """
    Calculate the percentage of fk when is 1 in a dataframe.
    :param obj: input dataframe.
    :return: percentage of fk when is 1.
    """
    # get frequency of keyVars
    fk = kAnonCalc(obj=obj)
    return (fk == 1).sum() * 100 / len(fk)

