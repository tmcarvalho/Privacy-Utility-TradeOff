import recordlinkage
from pandas.api.types import is_numeric_dtype
import warnings


def recordLinkage(transfObj, origObj, block):
    """
    Compare the transformed dataframe with the original dataframe.
    :param transfObj: transformed dataframe.
    :param origObj: initial dataframe.
    :param block: column to block.
    :return: all possible combinations.
    """
    return RL(transfObj=transfObj, origObj=origObj, block=block).verify_errors()


class RL:
    def __init__(self, transfObj, origObj, block):
        self.transfObj = transfObj
        self.origObj = origObj
        self.block = block

    def verify_errors(self):
        cols_orig = self.origObj.columns.values
        cols_transf = self.transfObj.columns.values
        if cols_transf not in cols_orig:
            warnings.warn("Variables in both datasets does not match!")
        elif self.block == "":
            warnings.warn("Empty block column!")
        else:
            return self.RLwork(cols_transf, cols_orig)

    def RLwork(self, cols_transf, cols_orig):
        indexer = recordlinkage.Index()
        # indexer.full()
        indexer.block(left_on=self.block, right_on=self.block)
        candidates = indexer.index(self.transfObj, self.origObj)
        # print(len(candidates))
        compare = recordlinkage.Compare()
        for i in range(0, len(cols_transf)):
            if is_numeric_dtype(self.transfObj[cols_transf[i]]):
                compare.numeric(cols_transf[i], cols_orig[i], label=cols_transf[i])
            else:
                compare.string(cols_transf[i], cols_orig[i], threshold=0.9, label=cols_transf[i])

        linkage = compare.compute(candidates, self.transfObj, self.origObj)
        return linkage


def calcRL(transfObj, origObj, block):
    """
    Calculate the percentage of combinations from the record linkage that have a high score.
    :param transfObj: transformed dataframe.
    :param origObj: initial dataframe.
    :param block: column to block.
    :return: percentage of combinations that have a high score.
    """
    rl = recordLinkage(transfObj, origObj, block)
    potential_matches = rl[rl.sum(axis=1) > 1].reset_index()
    potential_matches['Score'] = potential_matches.loc[:, potential_matches.columns[0]:potential_matches.columns[
        len(potential_matches.columns) - 1]].sum(axis=1)
    potential_matches = potential_matches[potential_matches['Score'] >= int((len(potential_matches.columns)) * 0.7)]
    freqs = potential_matches.groupby('level_0').size().reset_index(name='Count')
    max_risk = len(freqs[freqs.Count == 1])
    return (max_risk * 100) / len(transfObj)


