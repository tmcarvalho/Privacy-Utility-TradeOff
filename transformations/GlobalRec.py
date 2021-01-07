import pandas as pd
import numpy as np
import itertools
from ReIdentificationRisk import kAnon
import warnings


def globalRecoding(obj):
    """
    Global recoding of numerical (continuous) variables.
    :param obj: input dataframe.
    :return: global recoding dataframe.
    """
    return GlobalRec(obj=obj).verify_errors()


class GlobalRec:
    def __init__(self, obj):
        self.obj = obj

    def verify_errors(self):
        # get columns whose data type is int
        keyVars = self.obj.select_dtypes(include=np.int).columns
        if len(keyVars) == 0:
            raise ValueError("No variables for global recoding. Only integer type is acceptable!\n")
        else:
            return self.globalRec(keyVars)

    def find_range(self, keyVars):
        all_ranges = [0, 1, 2]
        for j in range(0, len(keyVars)):
            sigma = np.std(keyVars[j])
            all_ranges = [i * int(sigma) for i in all_ranges]
        return all_ranges

    def globalRec(self, keyVars):
        gen_fk = pd.DataFrame(columns=['fk_per_comb'])
        df_gen = self.obj.copy()
        ranges = self.find_range(keyVars)
        comb = [i for i in itertools.product(*ranges)]
        for index, x in enumerate(comb):
            if index == 0:
                dif = [comb[index][len(comb[0]) - 1]]
            elif (index == 0) & (index != len(comb) - 1):
                dif = list(set(comb[1]) - set(comb[0]))
            else:
                dif = [comb[index][0]]
            if (len(dif) != 0) & (len(dif) < 2):
                def has_equal_element(list1, list2):
                    return [e1 == e2 for e1, e2 in zip(list1, list2)]

                pos = [i for i, val in enumerate(has_equal_element(comb[index], comb[index - 1])) if val == False]
                if len(pos) == 0:
                    pass
                elif x[pos[0]] == 0:
                    df_gen[keyVars[pos[0] - 1]] = self.obj[keyVars[pos[0] - 1]]
                else:
                    df_gen[keyVars[pos[0] - 1]] = pd.cut(self.obj[keyVars[pos[0] - 1]],
                                                         bins=list(range(
                                                             min(self.obj[keyVars[pos[0] - 1]]) - x[
                                                                 pos[0]],
                                                             max(self.obj[keyVars[pos[0] - 1]]) + x[
                                                                 pos[0]],
                                                             x[pos[0]])))
                gen_fk.loc[index] = kAnon.calc_max_risk(df_gen)

        if gen_fk['fk_per_comb'].min() == 100:
            warnings.warn("Dataframe is at max risk!")
        elif gen_fk['fk_per_comb'].min() == 0:
            warnings.warn("Dataframe does not have observations with max risk!")
        idx_min = np.argmin(gen_fk['fk_per_comb'].values)
        comb_idx = comb[idx_min]
        for i in range(0, len(comb_idx)):
            if comb_idx[i] == 0:
                pass
            else:
                self.obj[keyVars[i]] = pd.cut(self.obj[keyVars[i]],
                                              bins=list(range(min(self.obj[keyVars[i]]) - comb_idx[i],
                                                              max(self.obj[keyVars[i]]) + comb_idx[i],
                                                              comb_idx[i])))
        return self.obj
