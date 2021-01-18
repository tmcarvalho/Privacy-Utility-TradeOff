from ReIdentificationRisk import kAnon, RecordLinkage


def calc_max_fk(obj):
    """
    Calculate the percentage of fk when is 1 in a dataframe.
    :param obj: input dataframe.
    :return: percentage of fk when is 1.
    """
    # get frequency of keyVars
    fk = kAnon.kAnonCalc(obj)
    return (fk == 1).sum() * 100 / len(fk)


def calc_max_rl(obj, origObj, block, indexer):
    """
    Calculate the percentage of combinations from the record linkage that have a high score.
    :param transfObj: transformed dataframe.
    :param origObj: initial dataframe.
    :param block: column to block.
    :return: percentage of combinations that have a high score.
    """
    rl = RecordLinkage.recordLinkage(obj, origObj, block, indexer)
    freqs = rl.groupby('level_0').size().reset_index(name='Count')
    max_risk = len(freqs[freqs.Count == 1])

    return (max_risk * 100) / len(obj)


def relative_error(keyVar_origObj, keyVar_transfObj):
    """
    Calculate the relative error to find the best epsilon in the noise addition.
    :param keyVar_origObj: variable from original dataframe.
    :param keyVar_transfObj: variable from transformed dataframe
    :return: relative error for each value
    """
    # |(protected value - original value)| / original value
    error = ((keyVar_transfObj - keyVar_origObj).abs() / keyVar_origObj)
    return error


