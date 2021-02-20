import recordlinkage
from pandas.api.types import is_numeric_dtype
import warnings


def recordLinkage(transfObj, origObj, block, rlIndexer):
    """
    Compare the transformed dataframe with the original dataframe.
    :param transfObj: transformed dataframe.
    :param origObj: initial dataframe.
    :param block: column to block.
    :param rlIndexer: define type of indexer (full or block).
    :return: potential matches between the two datasets.
    """
    return RL(transfObj=transfObj, origObj=origObj, block=block, rlIndexer=rlIndexer).verify_errors()


class RL:
    def __init__(self, transfObj, origObj, block, rlIndexer):
        self.transfObj = transfObj
        self.origObj = origObj
        self.block = block
        self.rlIndexer = rlIndexer

    def verify_errors(self):
        cols_orig = self.origObj.columns.values
        cols_transf = self.transfObj.columns.values
        if cols_transf not in cols_orig:
            warnings.warn("Variables in both datasets does not match!")
        elif (self.block == "") and (self.rlIndexer == "block"):
            warnings.warn("Empty block column!")
        elif self.rlIndexer == "":
            warnings.warn("indexer should be specified as 'full' or 'block'!")
        else:
            return self.RLwork(cols_transf, cols_orig)

    def RLwork(self, cols_transf, cols_orig):
        indexer = recordlinkage.Index()
        # if self.rlIndexer == 'full':
        # indexer.full()
        # else:
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

        potential_matches = linkage[linkage.sum(axis=1) > 1].reset_index()
        potential_matches['Score'] = potential_matches.loc[:, potential_matches.columns[0]:potential_matches.columns[
            len(potential_matches.columns) - 1]].sum(axis=1)
        potential_matches = potential_matches[potential_matches['Score'] >= int((len(potential_matches.columns)) * 0.7)]

        return potential_matches



