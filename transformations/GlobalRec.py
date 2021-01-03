import pandas as pd
import numpy as np
import itertools
from ReIdentificationRisk import kAnon
import warnings


def globalRecoding(obj, keyVars):
    """
    Global recoding of numerical (continuous) variables.
    :param obj: input data.
    :param keyVars: names (or indices) of key variables.
    :return: global recoding dataframe.
    """
    return GlobalRec(obj=obj, keyVars=keyVars).verify_errors()


class GlobalRec:
    def __init__(self, obj, keyVars):
        self.obj = obj
        self.keyVars = keyVars

    def verify_errors(self):
        # get columns whose data type is int
        filteredCols = self.obj.select_dtypes(include=np.int).columns.values
        # get the columns that are not in dataframe
        columns = list(self.obj.columns.values)
        error_vars = np.setdiff1d(self.keyVars, columns)
        if len(error_vars) != 0:
            raise ValueError("[" + '%s' % ', '.join(map(str, error_vars)) + "] specified in 'keyVars' can "
                                                                            "not be found!\n")
        elif not all(x in filteredCols for x in self.keyVars):
            raise ValueError("There is a column specified in 'keyVars' that is not integer type!\n")
        else:
            return self.globalRec()

    def find_range(self):
        all_ranges = []
        for j in range(0, len(self.keyVars)):
            # define ranges size according to the max of a variable
            if self.obj[self.keyVars[j]].max() <= 10:
                ranges = [0, 2, 4]
                ranges = [r for r in ranges if (r < self.obj[self.keyVars[j]].max())]
            elif self.obj[self.keyVars[j]].max() > 10:
                ranges = [0, 4, 7, 10, 12]
                ranges = [r for r in ranges if (r < self.obj[self.keyVars[j]].max())]
            all_ranges.append(ranges)
        return all_ranges

    def globalRec(self):
        gen_fk = pd.DataFrame(columns=['fk_per_comb'])
        df_gen = self.obj.copy()
        ranges = self.find_range()
        comb = [i for i in itertools.product(*ranges)]
        dif = []
        if len(ranges) == 1:
            raise warnings.warn("All selected 'keyVars' has maximum less than 2!")
            pass
        else:
            for index, x in enumerate(comb):
                if index == 0:
                    pass
                elif (index != 0) & (index != len(comb) - 1):
                    dif = list(set(comb[index]) - set(comb[index - 1]))
                else:
                    dif = [max(comb[index])]
                if (len(dif) != 0) & (len(dif) < 2):
                    def has_equal_element(list1, list2):
                        return [e1 == e2 for e1, e2 in zip(list1, list2)]

                    pos = [i for i, val in enumerate(has_equal_element(comb[index], comb[index - 1])) if val == False]
                    if len(pos) == 0:
                        pass
                    elif x[pos[0]] == 0:
                        df_gen[self.keyVars[pos[0] - 1]] = self.obj[self.keyVars[pos[0] - 1]]
                    else:
                        df_gen[self.keyVars[pos[0] - 1]] = pd.cut(self.obj[self.keyVars[pos[0] - 1]],
                                                                  bins=list(range(
                                                                      min(self.obj[self.keyVars[pos[0] - 1]]) - x[
                                                                          pos[0]],
                                                                      max(self.obj[self.keyVars[pos[0] - 1]]) + x[
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
                self.obj[self.keyVars[i]] = pd.cut(self.obj[self.keyVars[i]],
                                                   bins=list(range(min(self.obj[self.keyVars[i]]) - comb_idx[i],
                                                                   max(self.obj[self.keyVars[i]]) + comb_idx[i],
                                                                   comb_idx[i])))
        return self.obj

