import pandas as pd
import numpy as np
import itertools
from ReIdentificationRisk import CalcRisk
import warnings


def globalRecoding(obj, std_magnitude=1):
    """
    Global recoding of numerical (continuous) variables.
    :param obj: input dataframe.
    :param std_magnitude: standard deviation magnitude to define the binning size.
    :return: best combination and the respective keyVars.
    """
    return GlobalRec(obj=obj, std_magnitude=std_magnitude).verify_errors()


class GlobalRec:
    def __init__(self, obj, std_magnitude):
        self.obj = obj
        self.std_magnitude = std_magnitude
        self.keyVars = self.obj.select_dtypes(include=np.int).columns

    def verify_errors(self):
        if len(self.keyVars) == 0:
            # warnings.warn("No variables for global recoding. Only integer type is acceptable!\n")
            return self.obj
        else:
            return self.globalRec()

    def find_range(self):
        all_ranges = []
        # magnitude = [0.5, 1.5]
        for j, var in enumerate(self.keyVars):
            # define bin size with std and magnitude of std
            sigma = np.std(self.obj[self.keyVars[j]])
            if isinstance(self.std_magnitude, list):
                ranges = [int(i * sigma) for i in self.std_magnitude]
            else:
                ranges = int(self.std_magnitude * sigma)
            # remove variables with no global recoding
            if (isinstance(ranges, list)) and (all(a != 0 for a in ranges)):
                all_ranges.append(ranges)
            elif not isinstance(ranges, list):
                all_ranges.append(ranges)
            else:
                self.keyVars.remove(var)

        return all_ranges

    def globalRec(self):
        # risk = pd.DataFrame(columns=['fk_per_comb', 'rl_per_comb'])
        risk = pd.DataFrame(columns=['fk_per_comb'])
        df_gen = self.obj.copy()
        dif = []
        comb = []
        # remove columns that have only two unique
        # vars_two_uniques = self.obj.loc[:, self.obj.apply(pd.Series.nunique) == 2].columns
        # self.keyVars = list(set(self.keyVars) - set(vars_two_uniques))
        if len(self.keyVars) != 0:
            # bin size for each column
            ranges = self.find_range()
            if isinstance(self.std_magnitude, list):
                comb = [i for i in itertools.product(*ranges)]
                print('GlobalRec combs: ' + str(len(comb)))
                for index, x in enumerate(comb):
                    if all(v == 0 for v in x):
                        pass
                    elif (index != 0) & (index != len(comb) - 1):
                        # get the value that is different in two subsequents combinations
                        dif = list(set(comb[index]) - set(comb[index - 1]))
                    else:
                        dif = [comb[index][0]]
                    if len(dif) == 1:
                        def has_equal_element(list1, list2):
                            return [e1 == e2 for e1, e2 in zip(list1, list2)]

                        # get the position of the value that is different
                        pos = [i for i, val in enumerate(has_equal_element(comb[index], comb[index - 1])) if val == False]
                        if len(pos) == 0:
                            pass
                        elif x[pos[0]] == 0:
                            # when bin size is zero, the column remains with original values
                            df_gen[self.keyVars[pos[0]]] = self.obj[self.keyVars[pos[0]]]
                        else:
                            bins = list(range(min(self.obj[self.keyVars[pos[0]]]),
                                              max(self.obj[self.keyVars[pos[0]]]) + x[pos[0]], x[pos[0]]))
                            labels = ['%d' % bins[i] for i in range(0, len(bins) - 1)]
                            df_gen[self.keyVars[pos[0]]] = pd.cut(self.obj[self.keyVars[pos[0]]], bins=bins, labels=labels,
                                                                  include_lowest=True).astype(int)

                        risk.loc[index, 'fk_per_comb'] = CalcRisk.calc_max_fk(df_gen.select_dtypes(exclude=np.float))
                        # risk.loc[index, 'rl_per_comb'] = RecordLinkage.calcRL(df_gen, self.obj)
                if len(risk) != 0:
                    # get the combination with minimum re-identification risk
                    minimum = risk['fk_per_comb'].min()
                    idx_min = risk[risk['fk_per_comb'] == minimum].index
                    # get the best combination
                    comb_idx = comb[idx_min[0]]
                    return self.best_bin_size(comb_idx)
                else:
                    return self.obj

            else:
                return self.best_bin_size(ranges)

    def best_bin_size(self, comb_idx):
        # assign best combination of bin size to the dataset
        for i in range(0, len(comb_idx)):
            if (all(self.obj[self.keyVars[i]] != '*')) and (self.obj[self.keyVars[i]].nunique() != 1):
                if comb_idx[i] == 0:
                    pass
                else:
                    bins = list(range(min(self.obj[self.keyVars[i]]),
                                      max(self.obj[self.keyVars[i]]) + comb_idx[i],
                                      comb_idx[i]))
                    labels = ['%d' % bins[i] for i in range(0, len(bins) - 1)]
                    self.obj[self.keyVars[i]] = pd.cut(self.obj[self.keyVars[i]], bins=bins, labels=labels,
                                                       include_lowest=True).astype(int)
        return self.obj
