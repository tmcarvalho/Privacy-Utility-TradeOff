import numpy as np
import warnings


def suppression(obj):
    """
    Suppression of columns which have a number of unique values above 90% excluding floating points.
    :param obj: input dataframe.
    :return: suppressed dataframe or original dataframe if it does not have high percentage of unique values.
    """
    return Sup(obj=obj).verify_errors()


class Sup:
    def __init__(self, obj):
        self.obj = obj

    def verify_errors(self):
        if len(self.obj.select_dtypes(exclude=np.float).columns) == 0:
            # warnings.warn("Dataframe does not have any variable to suppress!")
            return self.obj
        else:
            return self.supWork()

    def supWork(self):
        # percentage of uniques in all variables except floating points
        uniques_per = self.obj.select_dtypes(exclude=np.float).apply(lambda col: col.nunique() * 100 / len(self.obj))
        # define maximum percentage
        uniques_max_per = uniques_per[uniques_per > 90]
        if len(uniques_max_per) != 0:
            # list of columns to suppress
            cols = self.obj.columns[self.obj.columns.isin(uniques_max_per.index)].values
            # create key : scalar value dictionary
            scalar_dict = {c: '*' for c in cols}
            # assign columns with '*' which represents the suppression
            self.obj = self.obj.assign(**scalar_dict)
        # else:
            # warnings.warn("None of variables has unique values above 90%!")
        return self.obj
