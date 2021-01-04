import numpy as np
import warnings


def suppression(obj):
    """
    Suppression of columns which have a number of uniques values above 90%.
    :param obj: input data.
    :return: suppressed dataframe or original dataframe.
    """
    return Sup(obj=obj).verify_errors()


class Sup:
    def __init__(self, obj):
        self.obj = obj

    def verify_errors(self):
        if len(self.obj.select_dtypes(exclude=np.float).columns) == 0:
            warnings.warn("Dataframe does not have any variable to suppress!")
        return self.supWork()

    def supWork(self):
        # get percentage of uniques in all variables except floating points
        uniques_per = self.obj.select_dtypes(exclude=np.float).apply(lambda col: col.nunique() * 100 / len(self.obj))
        # define maximum percentage
        uniques_max_per = uniques_per[uniques_per > 90]
        if len(uniques_max_per) != 0:
            # assign columns with '*' to represent the suppression
            self.obj.loc[:, self.obj.columns.isin(uniques_max_per.index)] = '*'
        else:
            warnings.warn("None of variables has unique values above 90%!")
        return self.obj
