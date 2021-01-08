import recordlinkage
from pandas.api.types import is_numeric_dtype


def recordLinkage(transfObj, origObj):
    """

    :param transfObj:
    :param origObj:
    :return:
    """
    return RL(transfObj=transfObj, origObj=origObj).verify_errors()


class RL:
    def __init__(self, transfObj, origObj):
        self.transfObj = transfObj
        self.origObj = origObj

    def verify_errors(self):
        cols_orig = self.origObj.columns.values
        cols_transf = self.transfObj.columns.values
        if cols_transf not in cols_orig:
            raise ValueError("Variables in both datasets does not match!")
        else:
            return self.RLwork(cols_transf, cols_orig)

    def RLwork(self, cols_transf, cols_orig):
        indexer = recordlinkage.Index()
        indexer.full()
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


def calcRL(transfObj, origObj):
    rl = recordLinkage(transfObj, origObj)
    potential_matches = rl[rl.sum(axis=1) > 1].reset_index()
    potential_matches['Score'] = potential_matches.loc[:, potential_matches.columns[0]:potential_matches.columns[
        len(potential_matches.columns) - 1]].sum(axis=1)
    potential_matches = potential_matches[potential_matches['Score'] >= int((len(potential_matches.columns)) * 0.7)]
    freqs = potential_matches.groupby('level_0').size().reset_index(name='Count')
    max_risk = len(freqs[freqs.Count == 1])
    return (max_risk * 100) / len(transfObj)


