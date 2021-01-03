from pandas.api.types import is_numeric_dtype


def topBottomCoding(obj, value, replacement, kind="top", column=None):
    """
    Replace extreme values, larger or lower than a threshold, by a different value.
    :param obj: input data.
    :param value: value that will be top or bottom coded
    :param replacement: replacement value
    :param kind: top or bottom
    :param column: variable name
    :return: top or bottom coded data.
    """
    return TopBot(obj=obj, value=value, replacement=replacement, kind=kind, column=column).verify_errors()


class TopBot:
    def __init__(self, obj, value, replacement, kind, column):
        self.x = obj[[column]]
        self.obj = obj
        self.value = value
        self.replacement = replacement
        self.kind = kind
        self.column = column

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

