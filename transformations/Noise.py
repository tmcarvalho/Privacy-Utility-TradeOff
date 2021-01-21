from pandas.api.types import is_float_dtype
import warnings
import pandas as pd
from ReIdentificationRisk import CalcRisk
import numpy as np


def addNoise(obj):
    """
    Add random noise with Laplace mechanism.
    :param obj: input dataframe.
    :return: relative error for each tested epsilon.
    """
    return Noise(obj=obj).verify_errors()


class Noise:
    def __init__(self, obj):
        self.obj = obj

    def verify_errors(self):
        # get columns whose data type is float
        keyVars = self.obj.select_dtypes(include=np.float).columns
        if len(keyVars) == 0:
            warnings.warn("No variables for adding noise. Only integer type is acceptable!\n")
            return self.obj
        else:
            return self.noiseWork(keyVars)

    def noiseWork(self, keyVars):
        ep = [0.5, 2, 4, 8]
        # dataframe to store relative error
        relative_error = pd.DataFrame()
        df_noise = self.obj.copy()
        for col in keyVars:
            diam = df_noise[col].max() - df_noise[col].min()
            if int(diam) == 0:
                warnings.warn("Diameter of the variable is 0! No noise to be applied!\n")
                pass
            else:
                for i in range(0, len(ep)):
                    if is_float_dtype(df_noise[col]):
                        df_noise[col] = df_noise[col].apply(lambda x: x + Laplace(diam, ep[i])).astype(float)
                    else:
                        df_noise[col] = df_noise[col].apply(lambda x: x + Laplace(diam, ep[i])).astype(float)

                    rel_error = CalcRisk.relative_error(self.obj[col], df_noise[col])
                    # replace inf values caused by denominator = zero
                    if (min(rel_error) == np.inf) and (max(rel_error) == np.inf):
                        relative_error.loc[i, col] = 0
                    elif max(rel_error) == np.inf:
                        rel_error.replace([np.inf], max(rel_error.replace(np.inf, np.nan)), inplace=True)
                        relative_error.loc[i, col] = np.mean(rel_error)

        return relative_error


def Laplace(diam, ep):
    return np.random.laplace(0, diam/ep, 1)[0]


def assign_best_ep(relative_error, df):
    ep = [0.5, 2, 4, 8]
    # assign best epsilon to the dataset
    keyVars = relative_error.columns
    if len(keyVars) == 0:
        warnings.warn("Noise cannot be applied because all columns have diameter zero!")
    else:
        for col in keyVars:
            diam = df[col].max() - df[col].min()
            min_val = min((abs(x), x) for x in relative_error[col])[1]
            idx = relative_error[col].tolist().index(min_val)
            if is_float_dtype(df[col]):
                df[col] = df[col].apply(lambda x: x + Laplace(diam, ep[idx]))
                df[col] = df[col].apply(lambda x: format(x, '.2f')).astype(float)
            else:
                df[col] = df[col].apply(lambda x: x + Laplace(diam, ep[idx]))
                df[col] = df[col].apply(lambda x: format(x, '.0f')).astype(int)

    return df

