import numpy as np
from pandas.api.types import is_float_dtype


def addNoise(obj):
    """

    :param obj:
    :return:
    """
    return Noise(obj=obj).verify_errors()


class Noise:
    def __init__(self, obj):
        self.obj = obj

    def verify_errors(self):
        # get columns whose data type is int
        keyVars = self.obj.select_dtypes(include=np.number).columns
        if len(keyVars) == 0:
            raise ValueError("No variables for adding noise. Only integer type is acceptable!\n")
        else:
            return self.noiseWork(keyVars)

    def noiseWork(self, keyVars):
        ep = 0.5
        df_noise = self.obj.copy()
        for col in keyVars:
            laplace = self.Laplace(ep)
            if is_float_dtype(df_noise[col]):
                print(col)
                df_noise[col] = df_noise[col] + laplace
                df_noise[col] = df_noise[col].apply(lambda x: format(x, '.2f')).astype(float)
            else:
                df_noise[col] = df_noise[col] + laplace
                df_noise[col] = df_noise[col].apply(lambda x: format(x, '.0f')).astype(int)
            # df[col] = df[col].astype('float64')
        return df_noise

    def Laplace(self, ep):
        sensitivity = 1
        beta = sensitivity / ep
        # Gets random laplacian noise for all values
        laplacian_noise = np.random.laplace(0, beta, 1)

        return laplacian_noise
