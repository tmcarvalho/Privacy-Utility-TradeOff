import warnings
import numpy as np
import pandas as pd
from ReIdentificationRisk import CalcRisk


def topBottomCoding(obj, origObj):
    """
    Replace extreme values, larger or lower than a threshold, by a different value.
    :param obj: input dataframe.
    :param origObj: original dataframe to apply record linkage.
    :param outlier: inner or outer fence values to find outliers.
    :return: top or bottom coded data.
    """
    return TopBot(obj=obj, origObj=origObj, outlier=outlier).verify_errors()


class TopBot:
    def __init__(self, obj, origObj, outlier):
        self.obj = obj
        self.origObj = origObj
        self.outlier = outlier
        self.keyVars = self.obj.select_dtypes(include=np.number).columns

    def verify_errors(self):
        if len(self.keyVars) == 0:
            # warnings.warn("Dataframe does not have any variable to apply Top & Bottom!")
            return self.obj
        else:
            return self.topBotWork()

    def topBotWork(self):
        risk = pd.DataFrame(columns=['rl_per_comb'])
        # outer_fence = [1.5, 3]
        data_to_transform = self.obj.copy()
        df_org = self.origObj.copy()
        if isinstance(self.outlier, list):
            for of in self.outlier:
                for j in range(0, len(self.keyVars)):
                    if all(self.obj[self.keyVars[j]]) != '*':
                        # outliers detection with Tukey's method
                        out_prob, outer_le, outer_ue = self.tukeys_method(self.keyVars[j], of)
                        if len(out_prob) != 0:
                            if data_to_transform[self.keyVars[j]].dtype == np.int:
                                data_to_transform.loc[data_to_transform[self.keyVars[j]] <= outer_le, self.keyVars[j]] = int(
                                    outer_le)
                                data_to_transform.loc[data_to_transform[self.keyVars[j]] >= outer_ue, self.keyVars[j]] = int(
                                    outer_ue)
                            else:
                                data_to_transform.loc[
                                    data_to_transform[self.keyVars[j]] <= outer_le, self.keyVars[j]] = outer_le
                                data_to_transform.loc[
                                    data_to_transform[self.keyVars[j]] >= outer_ue, self.keyVars[j]] = outer_ue

                check_types = df_org.dtypes.eq('object') == data_to_transform.dtypes.eq('object')
                idx = np.where(check_types == False)[0]
                if len(idx) >= 1:
                    cols = df_org.columns[idx]
                    for col in cols:
                        df_org[col] = df_org[col].astype(str)

                # limit record linkage with blocking
                max_unique = max(data_to_transform.nunique())
                # avoid too many candidates to record linkage
                max_unique1 = list(set(data_to_transform.nunique()))
                if (max_unique >= len(data_to_transform) * 0.85) and (len(max_unique1) != 1):
                    max_unique1 = sorted(max_unique1)
                    idx = [i for i, v in enumerate(max_unique1) if v < len(data_to_transform) * 0.85]
                    if len(idx) != 0:
                        block_column = data_to_transform.columns[data_to_transform.nunique() == max_unique1[idx[-1]]]
                    else:
                        block_column = data_to_transform.columns[data_to_transform.nunique() == max_unique1[0]]
                else:
                    block_column = data_to_transform.columns[data_to_transform.nunique() == max_unique]

                risk.loc[of, 'rl_per_comb'] = CalcRisk.calc_max_rl(data_to_transform, df_org, block_column[0],
                                                                  indexer="block")
                data_to_transform = self.obj.copy()
                df_org = self.origObj.copy()

            risk = risk.reset_index(drop=False)
            min = risk['rl_per_comb'].min()
            outer_fence = risk['index'][risk['rl_per_comb'] == min].min()
            return self.best_outer_fence(outer_fence)

        else:
            return self.best_outer_fence(self.outlier)

    def tukeys_method(self, keyVar, of):
        q1 = self.obj[keyVar].quantile(0.25)
        q3 = self.obj[keyVar].quantile(0.75)
        iqr = q3 - q1
        outer_fence = of * iqr

        # outer fence lower and upper end
        outer_fence_le = q1 - outer_fence
        outer_fence_ue = q3 + outer_fence

        outliers_prob = []
        for index, x in enumerate(self.obj[keyVar]):
            if (x <= outer_fence_le or x >= outer_fence_ue) and outer_fence_le != outer_fence_ue:
                outliers_prob.append(index)

        return outliers_prob, outer_fence_le, outer_fence_ue

    def best_outer_fence(self, outer_fence):
        for j in range(0, len(self.keyVars)):
            if all(self.obj[self.keyVars[j]] != '*'):
                # outliers detection with Tukey's method
                out_prob, outer_le, outer_ue = self.tukeys_method(self.keyVars[j], outer_fence)
                if len(out_prob) != 0:
                    if self.obj[self.keyVars[j]].dtype == np.int:
                        self.obj.loc[self.obj[self.keyVars[j]] <= outer_le, self.keyVars[j]] = int(outer_le)
                        self.obj.loc[self.obj[self.keyVars[j]] >= outer_ue, self.keyVars[j]] = int(outer_ue)
                    else:
                        self.obj.loc[self.obj[self.keyVars[j]] <= outer_le, self.keyVars[j]] = outer_le
                        self.obj.loc[self.obj[self.keyVars[j]] >= outer_ue, self.keyVars[j]] = outer_ue

        return self.obj