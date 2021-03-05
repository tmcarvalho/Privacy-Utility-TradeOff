import numpy as np
import pandas as pd
from ReIdentificationRisk import CalcRisk


def rounding(obj, origObj, base=5):
    """
    Round to specific base.
    :param obj: input dataframe.
    :param origObj: original dataframe to apply record linkage.
    :param base: rounding base.
    :return: dataframe with rounded bases.
    """
    return Rounding(obj=obj, origObj=origObj, base=base).verify_errors()


class Rounding:
    def __init__(self, obj, origObj, base):
        self.obj = obj
        self.origObj = origObj
        self.base = base
        self.keyVars = self.obj.select_dtypes(include=np.number).columns

    def verify_errors(self):
        if len(self.keyVars) == 0:
            # warnings.warn("Dataframe does not have any floating point to round!")
            return self.obj
        else:
            return self.roundWork()

    def roundWork(self):
        risk = pd.DataFrame(columns=['rl_per_comb'])
        # base = [0.2, 5, 10]
        df_round = self.obj.copy()
        df_org = self.origObj.copy()
        if isinstance(self.base, list):
            for b in self.base:
                for col in self.keyVars:
                    if all(df_round[col] != '*'):
                        df_round[col] = df_round[col].apply(lambda x: b * round(x/b))

                # if variable is float, with base=5 or base=10, the variable type will change to int
                # and record linkage will not work well
                df_rl = df_round.copy()
                for col in self.keyVars:
                    # assign with the same types
                    df_rl[col] = df_rl[col].astype(df_org[col].dtypes.name)

                # limit record linkage with blocking
                max_unique = max(df_rl.nunique())
                # avoid too many candidates to record linkage
                max_unique1 = list(set(df_rl.nunique()))
                if (max_unique >= len(df_rl) * 0.85) and (len(max_unique1) != 1):
                    max_unique1 = sorted(max_unique1)
                    idx = [i for i, v in enumerate(max_unique1) if v < len(df_rl) * 0.85]
                    if len(idx) != 0:
                        block_column = df_rl.columns[df_rl.nunique() == max_unique1[idx[-1]]]
                    else:
                        block_column = df_rl.columns[df_rl.nunique() == max_unique1[0]]
                else:
                    block_column = df_rl.columns[df_rl.nunique() == max_unique]
                if df_rl[block_column[0]].nunique() == 1:
                    risk.loc[b, 'rl_per_comb'] = 0
                else:
                    risk.loc[b, 'rl_per_comb'] = CalcRisk.calc_max_rl(df_rl, df_org, block_column[0], indexer="block")
                df_round = self.obj.copy()
                df_org = self.origObj.copy()

            risk = risk.reset_index(drop=False)
            min = risk['rl_per_comb'].min()
            b = risk['index'][risk['rl_per_comb'] == min].min()

            return self.best_base_round(b)

        else:
            return self.best_base_round(self.base)

    def best_base_round(self, base):
        obj_before = self.obj.copy()
        for col in self.keyVars:
            if all(self.obj[col] != '*'):
                self.obj[col] = self.obj[col].apply(lambda x: base * round(x / base))
                # assign with the same types
                self.obj[col] = self.obj[col].astype(obj_before[col].dtypes.name)

        return self.obj
