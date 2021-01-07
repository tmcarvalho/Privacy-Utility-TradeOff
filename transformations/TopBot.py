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
        if isinstance(self.column, list):
            raise ValueError("Length of argument 'column' > 1\n")
        if self.column not in self.obj.columns:
            raise ValueError("Variable specified in 'column' can not be found!\n")
        if not is_numeric_dtype(self.obj[self.column]):
            raise ValueError("Specified column is not numeric. topBottomCoding() can only be applied to numeric "
                             "variables!\n")
        elif is_numeric_dtype(self.obj[self.column]):
            self.obj[[self.column]] = self.topBotWork()
            return self.obj[[self.column]]

    def topBotWork(self):
        column_transformed = self.x.copy()
        if self.kind == "top":
            column_transformed.iloc[column_transformed > self.value] = self.replacement
        else:
            column_transformed.iloc[column_transformed < self.value] = self.replacement
        return column_transformed

        outliers_prob = []
        for index, x in enumerate(self.obj[keyVar]):
            if (x <= outer_fence_le or x >= outer_fence_ue) and outer_fence_le != outer_fence_ue:
                outliers_prob.append(index)

        return outliers_prob, outer_fence_le, outer_fence_ue
