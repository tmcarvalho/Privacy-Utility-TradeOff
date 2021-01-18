
def kAnonCalc(obj):
    """
    Measure the individual risk base on the frequency of the equivalence classes.
    :param obj: input dataframe.
    :return: frequency of each equivalence class for each observation.
    """
    return calcFreq(obj=obj).kAnonWork()


class calcFreq:
    def __init__(self, obj):
        self.obj = obj

    def kAnonWork(self):
        keyVars = list(self.obj.columns.values)
        fk = self.obj.groupby(keyVars)[keyVars[0]].transform(len)
        return fk




