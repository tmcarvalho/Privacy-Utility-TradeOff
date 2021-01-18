import psutil
from transformations import TopBot, GlobalRec, Suppression, RoundFloats, Noise
from ReIdentificationRisk import kAnon, RecordLinkage, CalcRisk
import numpy as np
import itertools
from decimal import Decimal
import pandas as pd
import time
import ray


# %% All functions to processing
def load_data():
    dss = np.load('DS_clean.npy', allow_pickle=True)
    ds = dss.copy()
    return ds


def change_cols_types(df):
    """
    Remove trailing zeros and assign as 'int' type.
    :param df: input data.
    :return: transformed data.
    """
    cols = df.select_dtypes(include=np.number).columns.values
    for col in cols:
        df[col] = df[col].apply(Decimal).astype(str)
        if any('.' in s for s in df[col]):
            df[col] = df[col].astype('float')
        else:
            df[col] = df[col].astype('int')
    return df


def define_combs(df):
    """
    Define combinations of transformation techniques base on dtypes of the dataframe.
    :param df: input dataframe
    :return: list o combinations.
    """
    comb_cat = comb_float = comb_int = []
    if len(np.where((df.dtypes == np.int) == True)[0]) != 0:
        comb_int = ['topbot', 'globalrec', 'sup']
    if len(np.where((df.dtypes == np.float) == True)[0]) != 0:
        comb_float = ['topbot', 'round', 'noise']
    if len(np.where(((df.dtypes == 'category') == True) | ((df.dtypes == 'object') == True))[0] |
           ((df.dtypes == np.int) == True)[0]) != 0:
        # percentage of uniques in all variables except floating points
        uniques_per = df.select_dtypes(exclude=np.float).apply(lambda col: col.nunique() * 100 / len(df))
        # define maximum percentage
        uniques_max_per = uniques_per[uniques_per > 90]
        if len(uniques_max_per) != 0:
            comb_cat = ['sup']
    transf_tech = list(set().union(comb_int, comb_float, comb_cat))
    # transf_tech = ['topbot', 'globalrec', 'sup', 'round', 'noise']
    comb = [c for i in range(len(transf_tech) + 1) for c in itertools.combinations(transf_tech, i)]
    return comb


def transformations(x, df_transf, c):
    """
    Apply transformation techniques to the dataframe.
    :param x: combination.
    :param df_transf: input dataframe
    :return: transformed dataframe
    """
    if 'sup' in x:
        df_transf = Suppression.suppression(df_transf)
    if 'topbot' in x:
        df_transf = TopBot.topBottomCoding(df_transf)
    if 'round' in x:
        df_transf = RoundFloats.roundFloats(df_transf)
    if 'noise' in x:
        df_transf = Noise.addNoise(df_transf)
    if 'globalrec' in x:
        c += 1
        keyVars = comb = []
        if c == 1:
            keyVars, comb = GlobalRec.globalRecoding(df_transf)
            if len(comb) != 0:
                df_transf = GlobalRec.best_bin_size(df_transf, keyVars, comb)
        else:
            if len(comb) != 0:
                df_transf = GlobalRec.best_bin_size(df_transf, keyVars, comb)

    return df_transf, c


# %%
# start Ray
# ray.init()

# @ray.remote
def process_single_df(df):
    # list to store the transformed dataframes
    transformations_combs = []
    # dataframe to store re-identification risk
    reID_risk = pd.DataFrame(columns=['initial_fk'])
    c = 0
    df = change_cols_types(df)
    # keep target variable aside
    tgt = df.iloc[:, -1]
    # dataframe without target variable to apply transformation techniques
    df_val = df[df.columns[:-1]]
    # create combinations adequate to the dataframe
    comb = define_combs(df_val)
    fk_var = 'fk_per'
    rl_var = 'rl_per'
    for index, x in enumerate(comb):
        df_transf = df_val.copy()
        if index == 0:
            # calculate initial re-identification risk with k-anonymity
            reID_risk.loc[index, 'initial_fk'] = CalcRisk.calc_max_fk(df_transf)
        else:
            # apply transformations
            df_transf, c = transformations(x, df_transf, c)
            if "sup" in x:
                # make sure that both datasets has equal dtypes to all columns
                check_types = df_val.dtypes == df_transf.dtypes
                idx = np.where(check_types == False)[0]
                if len(idx) != 0:
                    col = df_val.columns[idx[0]]
                    df_val[col] = df_val[col].astype(str)
                else:
                    continue
            # recalculate re-identification risk with k-anonymity
            reID_risk.loc[index, fk_var] = CalcRisk.calc_max_fk(df_transf)
            # limit record linkage with blocking!
            max_unique = max(df_transf.nunique())
            block_column = df_transf.columns[df_transf.nunique() == max_unique]
            # calculate record linkage
            reID_risk.loc[index, rl_var] = CalcRisk.calc_max_rl(df_transf, df_val, block_column[0],
                                                                indexer="block")

            # reset df_val because of dtype change due to the suppression
            df_val = df[df.columns[:-1]]
            # add target to the transformed dataset
            df_transf[tgt.name] = tgt.values
            # save all results
            transformations_combs.append(df_transf)

        print('Tech combs: ' + str(index) + '/' + str(len(comb)))

    return transformations_combs, reID_risk, comb


ds = load_data()
start_time = time.time()
result_ids = []
# for i in range(len(ds)):
#     result_ids.append(process_single_df.remote(ds[i]))

all_risk = []
all_combs = []
all_transf_combs = []

# for i in range(len(ds)):
#     transf_combs, risk, combs = ray.get(result_ids[i])
#     all_transf_combs.append(transf_combs)
#     all_risk.append(risk)
#     all_combs.append(combs)

# close Ray
# ray.shutdown()
