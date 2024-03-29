import warnings
import pandas as pd
from ReIdentificationRisk import CalcRisk
import numpy as np


def addNoise(obj, ep=0.5):
    """
    Add random noise with Laplace mechanism.
    :param obj: input dataframe.
    :param ep: epsilon to apply Laplace noise.
    :return: relative error for each tested epsilon.
    """
    return Noise(obj=obj, ep=ep).verify_errors()


class Noise:
    def __init__(self, obj, ep):
        self.obj = obj
        self.ep = ep
        self.keyVars = self.obj.select_dtypes(include=np.float).columns
        self.QIVars = self.obj.select_dtypes(include=[np.int, 'category']).columns

    def verify_errors(self):
        if len(self.keyVars) == 0:
            # warnings.warn("No variables for adding noise. Only float type is acceptable!\n")
            return self.obj
        else:
            return self.noiseWork()

    def noiseWork(self):
        # ep = [0.5, 2, 4, 8, 16]
        # dataframe to store relative error
        relative_error = pd.DataFrame()
        df_noise = self.obj.copy()
        if isinstance(self.ep, list):
            for col in self.keyVars:
                # equivalence classes
                eq = df_noise.groupby(self.QIVars)
                for _, df_group in eq:
                    diam = df_group[col].max() - df_group[col].min()
                    if diam == 0:
                        # warnings.warn("Diameter of the variable is 0! No noise to be applied!\n")
                        pass
                    else:
                        for row_index, _ in df_group.iterrows():
                            for i in range(0, len(self.ep)):
                                df_group[col][row_index] = df_group[col][row_index].apply(lambda x: x + self.Laplace(diam, self.ep[i])).astype(float)
                        rel_error = CalcRisk.relative_error(self.obj[col], df_noise[col])
                        # replace inf values caused by denominator = zero
                        if (min(rel_error) == np.inf) and (max(rel_error) == np.inf):
                            relative_error.loc[i, col] = 0
                        elif max(rel_error) == np.inf:
                            rel_error.replace([np.inf], max(rel_error.replace(np.inf, np.nan)), inplace=True)
                            relative_error.loc[i, col] = np.mean(rel_error)
                        else:
                            relative_error.loc[i, col] = np.mean(rel_error)        
                            
                                # diam = df_noise[col].max() - df_noise[col].min()                

            # assign best epsilon to the dataset
            vars = relative_error.columns
            if len(vars) != 0:
                for col in vars:
                    if all(self.obj[col] != '*'):
                        eq = self.obj.groupby(self.QIVars)
                        for _, df_group in eq:
                            diam = df_group[col].max() - df_group[col].min()
                            min_val = min((abs(x), x) for x in relative_error[col])[1]
                            idx = relative_error[col].tolist().index(min_val)
                            for row_index, _ in df_group.iterrows():
                                for i in range(0, len(self.ep)):
                                    self.obj[col][row_index] = self.obj[col][row_index].apply(lambda x: x + self.Laplace(diam, self.ep[idx])).astype(float)

            return self.obj

        else:
            if len(self.keyVars) != 0:
                for col in self.keyVars:
                    if all(self.obj[col] != '*'):
                        eq = self.obj.groupby(self.QIVars)
                        for _, df_group in eq:
                            diam = df_group[col].max() - df_group[col].min()
                            for row_index, _ in df_group.iterrows():
                                self.obj[col][row_index] = self.obj[col][row_index].apply(lambda x: x + self.Laplace(diam, self.ep)).astype(float)
            return self.obj

    def Laplace(self, diam, epsilon):
        return np.random.laplace(0, diam / epsilon, 1)[0]
