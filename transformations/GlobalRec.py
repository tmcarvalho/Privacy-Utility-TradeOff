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
            warnings.warn("No variables for global recoding. Only integer type is acceptable!\n")
            return self.obj
        else:
            return self.globalRec(keyVars)

    def find_range(self, keyVars):
        all_ranges = []
        magnitude = [0, 1, 2]
        for j in range(0, len(keyVars)):
            sigma = np.std(self.obj[keyVars[j]])
            ranges = [i * int(sigma) for i in magnitude]
            all_ranges.append(ranges)
        return all_ranges

    def globalRec(self, keyVars):
        # risk = pd.DataFrame(columns=['fk_per_comb', 'rl_per_comb'])
        risk = pd.DataFrame(columns=['fk_per_comb'])
        df_gen = self.obj.copy()
        ranges = self.find_range(keyVars)
        comb = [i for i in itertools.product(*ranges)]
        print('GlobalRec combs: ' + str(len(comb)))
        for index, x in enumerate(comb):
            if index == 0:
                dif = [comb[index][len(comb[0]) - 1]]
            elif (index != 0) & (index != len(comb) - 1):
                dif = list(set(comb[index]) - set(comb[index - 1]))
            else:
                dif = [comb[index][0]]
            if len(dif) == 1:
                def has_equal_element(list1, list2):
                    return [e1 == e2 for e1, e2 in zip(list1, list2)]

                pos = [i for i, val in enumerate(has_equal_element(comb[index], comb[index - 1])) if val == False]
                if len(pos) == 0:
                    pass
                elif x[pos[0]] == 0:
                    df_gen[keyVars[pos[0]]] = self.obj[keyVars[pos[0]]]
                else:
                    bins = list(range(min(self.obj[keyVars[pos[0]]]) - x[pos[0]],
                                      max(self.obj[keyVars[pos[0]]]) + x[pos[0]], x[pos[0]]))
                    labels = ['%d' % bins[i] for i in range(0, len(bins) - 1)]
                    df_gen[keyVars[pos[0]]] = pd.cut(self.obj[keyVars[pos[0]]], bins=bins, labels=labels).astype(int)

                risk.loc[index, 'fk_per_comb'] = kAnon.calc_max_risk(df_gen)
                # risk.loc[index, 'rl_per_comb'] = RecordLinkage.calcRL(df_gen, self.obj)

        if risk['fk_per_comb'].min() == 100:
            warnings.warn("Dataframe is at max risk!")
        elif risk['fk_per_comb'].min() == 0:
            warnings.warn("Dataframe does not have observations with max risk!")
        idx_min = np.argmin(risk['fk_per_comb'].values)
        comb_idx = comb[idx_min]
        for i in range(0, len(comb_idx)):
            if comb_idx[i] == 0:
                pass
            else:
                bins = list(range(min(self.obj[keyVars[i]]) - comb_idx[i],
                                  max(self.obj[keyVars[i]]) + comb_idx[i],
                                  comb_idx[i]))
                labels = ['%d' % bins[i] for i in range(0, len(bins) - 1)]
                self.obj[keyVars[i]] = pd.cut(self.obj[keyVars[i]], bins=bins, labels=labels).astype(int)
        return self.obj

