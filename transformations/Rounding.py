import numpy as np
import pandas as pd
from ReIdentificationRisk import CalcRisk


def rounding(obj, origObj):
    """
    Round to specific base.
    :param obj: input dataframe.
    :param origObj: original dataframe to apply record linkage.
    :return: dataframe with rounded bases.
    """
    return Rounding(obj=obj, origObj=origObj).verify_errors()


class Rounding:
    def __init__(self, obj, origObj):
        self.obj = obj
        self.origObj = origObj

    def verify_errors(self):
        keyVars = self.obj.select_dtypes(include=np.number).columns
        if len(keyVars) == 0:
            # warnings.warn("Dataframe does not have any floating point to round!")
            return self.obj
        else:
            return self.roundWork(keyVars)

    def roundWork(self, keyVars):
        risk = pd.DataFrame(columns=['rl_per_comb'])
        base = [0.2, 5, 10]
        df_round = self.obj.copy()
        df_org = self.origObj.copy()
        for b in base:
            for col in keyVars:
                df_round[col] = df_round[col].apply(lambda x: b * round(x/b))

            # if variable is float, with base=5 or base=10, the variable type will change to int
            # and record linkage will not work well
            df_rl = df_round.copy()
            for col in keyVars:
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
        base = risk['index'][risk['rl_per_comb'] == min].min()

        return base, keyVars


def best_base_round(obj, keyVars, base):
    obj_before = obj.copy()
    for col in keyVars:
        if all(obj[col] != '*'):
            obj[col] = obj[col].apply(lambda x: base * round(x / base))
            # assign with the same types
            obj[col] = obj[col].astype(obj_before[col].dtypes.name)

    return obj


