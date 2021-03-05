import warnings
import numpy as np


def roundFloats(obj, n_round):
    """
    Limit floats to two decimal points.
    :param obj: input dataframe.
    :param n_round: number to round.
    :return: dataframe with rounded floats.
    """
    return Round(obj=obj, n_round=n_round).verify_errors()


class Round:
    def __init__(self, obj, n_round):
        self.obj = obj
        self.n_round = n_round
        self.keyVars = self.obj.select_dtypes(include=np.float).columns

    def verify_errors(self):
        if len(self.keyVars) == 0:
            # warnings.warn("Dataframe does not have any floating point to round!")
            return self.obj
        else:
            return self.roundWork()

    def roundWork(self):
        for col in self.keyVars:
            self.obj[col] = round(self.obj[col], self.n_round)

        return self.obj
