from pandas.api.types import is_float_dtype
import warnings
import pandas as pd
from ReIdentificationRisk import CalcRisk
import numpy as np


def addNoise(obj):
    """
    Add random noise with Laplace mechanism.
    :param obj: input dataframe.
    :return: noisy data.
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
        relative_error = pd.DataFrame(columns=keyVars.tolist())
        for i in range(0, len(ep)):
            df_noise = self.obj.copy()
            for col in keyVars:
                diam = df_noise[col].max() - df_noise[col].min()
                if diam == 0:
                    warnings.warn("Diameter of the variable is 0! No noise to be applied!\n")
                else:
                    # laplace = self.Laplace(diam, ep[i])
                    pass
                if is_float_dtype(df_noise[col]):
                    df_noise[col] = df_noise[col].apply(lambda x: x + self.Laplace(diam, ep[i]))
                else:
                    df_noise[col] = df_noise[col].apply(lambda x: x + self.Laplace(diam, ep[i]))
                relative_error.loc[i, col] = np.mean(CalcRisk.relative_error(self.obj[col], df_noise[col]))

        # assign best epsilon to the dataset
        for col in keyVars:
            diam = self.obj[col].max() - self.obj[col].min()
            min_val = min((abs(x), x) for x in relative_error[col])[1]
            idx = relative_error[col].tolist().index(min_val)
            if diam == 0:
                raise warnings.warn("Diameter of the variable is 0! No noise to be applied!\n")
            else:
                pass
            if is_float_dtype(self.obj[col]):
                self.obj[col] = self.obj[col].apply(lambda x: x + self.Laplace(diam, ep[idx]))
                self.obj[col] = self.obj[col].apply(lambda x: format(x, '.2f')).astype(float)
            else:
                self.obj[col] = self.obj[col].apply(lambda x: x + self.Laplace(diam, ep[idx]))
                self.obj[col] = self.obj[col].apply(lambda x: format(x, '.0f')).astype(int)

        return self.obj

    def Laplace(self, diam, ep):
        return np.random.laplace(0, diam/ep, 1)[0]
