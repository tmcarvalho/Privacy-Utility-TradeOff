import warnings
import numpy as np


def topBottomCoding(obj):
    """
    Replace extreme values, larger or lower than a threshold, by a different value.
    :param obj: input dataframe.
    :return: top or bottom coded data.
    """
    return TopBot(obj=obj).verify_errors()


class TopBot:
    def __init__(self, obj):
        self.obj = obj
        self.keyVars = self.obj.select_dtypes(include=np.number).columns

    def verify_errors(self):
        if len(self.keyVars) == 0:
            warnings.warn("Dataframe does not have any variable to apply Top & Bottom!")
            return self.obj
        else:
            return self.topBotWork()

    def topBotWork(self):
        data_to_transform = self.obj.copy()
        for j in range(0, len(self.keyVars)):
            # outliers detection with Tukey's method
            out_prob, outer_le, outer_ue = self.tukeys_method(self.keyVars[j])
            if len(out_prob) != 0:
                if self.obj[self.keyVars[j]].dtype == np.int:
                    data_to_transform.loc[data_to_transform[self.keyVars[j]] <= outer_le, self.keyVars[j]] = int(outer_le)
                    data_to_transform.loc[data_to_transform[self.keyVars[j]] >= outer_ue, self.keyVars[j]] = int(outer_ue)
                else:
                    data_to_transform.loc[data_to_transform[self.keyVars[j]] <= outer_le, self.keyVars[j]] = outer_le
                    data_to_transform.loc[data_to_transform[self.keyVars[j]] >= outer_ue, self.keyVars[j]] = outer_ue

        return data_to_transform

    def tukeys_method(self, keyVar):
        q1 = self.obj[keyVar].quantile(0.25)
        q3 = self.obj[keyVar].quantile(0.75)
        iqr = q3 - q1
        outer_fence = 3 * iqr

        # outer fence lower and upper end
        outer_fence_le = q1 - outer_fence
        outer_fence_ue = q3 + outer_fence

        outliers_prob = []
        for index, x in enumerate(self.obj[keyVar]):
            if (x <= outer_fence_le or x >= outer_fence_ue) and outer_fence_le != outer_fence_ue:
                outliers_prob.append(index)

        return outliers_prob, outer_fence_le, outer_fence_ue

