import numpy as np
import pandas as pd
import warnings
from ReIdentificationRisk import CalcRisk


def suppression(obj, origObj, uniq_per=90):
    """
    Suppression of columns which have a number of unique values above 90% excluding floating points.
    :param obj: input dataframe.
    :param origObj: original dataframe to apply record linkage.
    :param uniq_per: percentage of distinct values.
    :return: suppressed dataframe or original dataframe if it does not have high percentage of unique values.
    """
    return Sup(obj=obj, origObj=origObj, uniq_per=uniq_per).supWork()


class Sup:
    def __init__(self, obj, origObj, uniq_per):
        self.obj = obj
        self.origObj = origObj
        self.uniq_per = uniq_per

    def supWork(self):
        risk = pd.DataFrame(columns=['rl_per_comb'])
        uniques_per = self.obj.apply(lambda col: col.nunique() * 100 / len(self.obj))
        # per = [70, 80, 90]
        df_sup = self.obj.copy()
        df_org = self.origObj.copy()

        def sup(p, df_sup, df_org):
            # define maximum percentage
            uniques_max_per = uniques_per[uniques_per > p]
            if len(uniques_max_per) != 0:
                # list of columns to suppress
                cols = df_sup.columns[df_sup.columns.isin(uniques_max_per.index)].values
                # create key : scalar value dictionary
                scalar_dict = {c: '*' for c in cols}
                # assign columns with '*' which represents the suppression
                df_sup = df_sup.assign(**scalar_dict)

                check_types = df_org.dtypes.eq('object') == df_sup.dtypes.eq('object')
                idx = np.where(check_types == False)[0]
                if len(idx) >= 1:
                    cols = df_org.columns[idx]
                    for col in cols:
                        df_org[col] = df_org[col].astype(str)

                # limit record linkage with blocking
                max_unique = max(df_sup.nunique())
                # avoid too many candidates to record linkage
                max_unique1 = list(set(df_sup.nunique()))
                if (max_unique >= len(df_sup) * 0.85) and (len(max_unique1) != 1):
                    max_unique1 = sorted(max_unique1)
                    idx = [i for i, v in enumerate(max_unique1) if v < len(df_sup) * 0.85]
                    if len(idx) != 0:
                        block_column = df_sup.columns[df_sup.nunique() == max_unique1[idx[-1]]]
                    else:
                        block_column = df_sup.columns[df_sup.nunique() == max_unique1[0]]
                else:
                    block_column = df_sup.columns[df_sup.nunique() == max_unique]
                if df_sup[block_column[0]].nunique() == 1:
                    risk.loc[p, 'rl_per_comb'] = 0
                else:
                    risk.loc[p, 'rl_per_comb'] = CalcRisk.calc_max_rl(df_sup, df_org, block_column[0],
                                                                      indexer="block")
            return risk

        if isinstance(self.uniq_per, list):
            for p in self.uniq_per:
                risk = sup(p, df_sup, df_org)
                df_sup = self.obj.copy()
                df_org = self.origObj.copy()
        else:
            risk = sup(self.uniq_per, df_sup, df_org)

        risk = risk.reset_index(drop=False)
        min = risk['rl_per_comb'].min()
        per = risk['index'][risk['rl_per_comb'] == min].max()

        return per


def best_per_sup(obj, per):
    """
    Apply best percentage for suppression.
    :param obj: input dataframe.
    :param per: percentage value.
    :return: 
    """
    # percentage of uniques in all variables except floating points
    uniques_per = obj.apply(lambda col: col.nunique() * 100 / len(obj))

    # define maximum percentage
    uniques_max_per = uniques_per[uniques_per > per]
    if len(uniques_max_per) != 0:
        # list of columns to suppress
        cols = obj.columns[obj.columns.isin(uniques_max_per.index)].values
        # create key : scalar value dictionary
        scalar_dict = {c: '*' for c in cols}
        # assign columns with '*' which represents the suppression
        obj = obj.assign(**scalar_dict)
    # else:
        # warnings.warn("None of variables has unique values above 90%!")
    return obj

