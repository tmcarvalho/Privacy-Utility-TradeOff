from pandas.api.types import is_float_dtype
import warnings
import pandas as pd
from ReIdentificationRisk import CalcRisk
import numpy as np


def addNoise(origObj, obj, epsilon):
    """
    Add noise with Laplace mechanism.
    :param origObj: dataframe original to calculate relative error.
    :param obj: dataframe that will be transformed.
    :param epsilon: epsilon value or list of values to apply Laplace noise.
    :return: noisy data.
    """
    return Noise(origObj=origObj, obj=obj, epsilon=epsilon).verify_errors()


class Noise:
    def __init__(self, origObj, obj, epsilon):
        self.origObj = origObj
        self.obj = obj
        self.epsilon = epsilon

    def verify_errors(self):
        # get columns whose data type is int and float
        keyVars = self.obj.select_dtypes(include=np.number).columns
        if len(keyVars) == 0:
            warnings.warn("No variables for adding noise. Only integer type is acceptable!\n")
        elif (not isinstance(self.epsilon, list)) and (self.epsilon == ""):
            warnings.warn("Define a value or a list of values to the epsilon!")
        elif (isinstance(self.epsilon, list)) and (len(self.epsilon) == 0):
            warnings.warn("List of epsilon values can not be empty!")
        else:
            return self.checkEpsilon(keyVars)

    def checkEpsilon(self, keyVars):
        # ep = [0.5, 2, 4, 8]
        relative_error = pd.DataFrame(columns=keyVars.tolist())

        def noiseWork(ep):
            df_noise = self.obj.copy()
            for col in keyVars:
                diam = df_noise[col].max() - df_noise[col].min()
                if diam == 0:
                    raise warnings.warn("Diameter of the variable is 0! No noise to be applied!\n")
                else:
                    # laplace = self.Laplace(diam, ep[i])
                    pass
                if is_float_dtype(df_noise[col]):
                    df_noise[col] = df_noise[col].apply(lambda x: x + self.Laplace(diam, ep))
                else:
                    df_noise[col] = df_noise[col].apply(lambda x: x + self.Laplace(diam, ep))
                relative_error.loc[i, col] = np.mean(CalcRisk.relative_error(self.origObj[col], df_noise[col]))
            return df_noise, relative_error

        if isinstance(self.epsilon, list):
            for i in range(0, len(self.epsilon)):
                df, relative_error = noiseWork(self.epsilon[i])
        else:
            df, relative_error = noiseWork(self.epsilon)
        # assign best epsilon to the dataset
        # df_noise = self.obj.copy()
        # for col in keyVars:
        #     diam = df_noise[col].max() - df_noise[col].min()
        #     min_val = min((abs(x), x) for x in rel_error[col])[1]
        #     idx = rel_error[col].tolist().index(min_val)
        #     if diam == 0:
        #         raise warnings.warn("Diameter of the variable is 0! No noise to be applied!\n")
        #     else:
        #         pass
        #     if is_float_dtype(df_noise[col]):
        #         df_noise[col] = df_noise[col].apply(lambda x: x + self.Laplace(diam, ep[idx]))
        #         df_noise[col] = df_noise[col].apply(lambda x: format(x, '.2f')).astype(float)
        #     else:
        #         df_noise[col] = df_noise[col].apply(lambda x: x + self.Laplace(diam, ep[idx]))
        #         df_noise[col] = df_noise[col].apply(lambda x: format(x, '.0f')).astype(int)

        return df, relative_error

    def Laplace(self, diam, ep):
        return np.random.laplace(0, diam/ep, 1)[0]


