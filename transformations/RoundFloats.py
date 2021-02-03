import warnings
import numpy as np


def roundFloats(obj):
    """
    Limit floats to two decimal points.
    :param obj: input dataframe.
    :return: dataframe with rounded floats.
    """
    return Round(obj=obj).verify_errors()


class Round:
    def __init__(self, obj):
        self.obj = obj

    def verify_errors(self):
        keyVars = self.obj.select_dtypes(include=np.float).columns
        if len(keyVars) == 0:
            # warnings.warn("Dataframe does not have any floating point to round!")
            return self.obj
        else:
            return self.roundWork(keyVars)

    def roundWork(self, keyVars):
        for col in keyVars:
            self.obj[col] = round(self.obj[col], 3)

        return self.obj
