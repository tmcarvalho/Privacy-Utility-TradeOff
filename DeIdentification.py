import warnings
from transformations import TopBot, GlobalRec, Suppression, Rounding, Noise, RoundFloats
from ReIdentificationRisk import CalcRisk
import numpy as np
import itertools
from decimal import Decimal
import pandas as pd
import ray
import gc


# %% All functions to processing
def load_data():
    dss = np.load('DS_clean1.npy', allow_pickle=True)
    ds = dss.copy()
    return ds


def change_cols_types(df):
    """
    Remove trailing zeros and assign as 'int' type.
    :param df: input dataset.
    :return: transformed dataset.
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
    Define combinations of transformation techniques for a dataframe base on variables types.
    :param df: input dataframe.
    :return: list o combinations.
    """
    comb_cat = comb_float = comb_int = []
    if len(np.where(((df.dtypes == 'category') == True) | ((df.dtypes == 'object') == True))[0]) != 0:
        comb_cat = ['sup']
    if len(np.where((df.dtypes == np.int) == True)[0]) != 0:
        comb_int = ['sup', 'topbot', 'globalrec', 'round']
    if len(np.where((df.dtypes == np.float) == True)[0]) != 0:
        comb_float = ['topbot', 'round', 'noise', 'sup']

    # union of all possible transformations techniques
    transf_tech = set().union(comb_cat, comb_int, comb_float)
    transf_tech = [x for x in transf_tech]
    # generate combinations of transformations techniques
    comb = [c for i in range(len(transf_tech) + 1) for c in itertools.combinations(transf_tech, i)]
    return comb


def GR_combination(df, i, index):
    """
    Finding global recoding combinations.
    :param df: input dataframe.
    :param i: index of dataframe.
    :param index: index of combination.
    :return: variables to apply global recoding and corresponding discretization size.
    """
    df_sup = df.copy()
    keyVars = comb = []
    uniques_per = df_sup.select_dtypes(exclude=np.float).apply(lambda col: col.nunique() * 100 / len(df_sup))
    # define maximum percentage
    uniques_max_per = uniques_per[uniques_per > 90]
    cols = df_sup.columns[df_sup.columns.isin(uniques_max_per.index)].values
    if len(cols) != 0:
        df_sup.drop(cols, axis=1, inplace=True)
    if len(df_sup.select_dtypes(include=np.int).columns) != 0:
        try:
            keyVars, comb = GlobalRec.globalRecoding(df_sup)
            pd.to_pickle(comb, 'Remote_results/GRcomb_' + str(i) + '_' + str(index) + '.pkl')
            # comb = pd.read_pickle('Remote_results/GRcomb_' + str(i) + '.pkl')
        except:
            pass

    return keyVars, comb


def parameters(x, obj, orgObj, i, params, index):
    """
    Collect the best parameters to apply the transformations.
    :param x: combination of techniques.
    :param obj: input dataset.
    :param orgObj: original object to compare with transformed dataset.
    :param i: index of dataset.
    :param params: parameters from previous iterations.
    :param index: index of combination.
    :return: list of parameters.
    """
    if 'sup' in x:
        per = Suppression.suppression(obj, orgObj, uniq_per=[0.7, 0.8, 0.9])
        params['per'] = per
    if 'topbot' in x:
        outer_fence, keyVarsT = TopBot.topBottomCoding(obj, orgObj)
        params['keyVarsT'] = keyVarsT
        params['outer_fence'] = outer_fence
    if 'round' in x:
        base, keyVarsR = Rounding.rounding(obj, orgObj)
        params['keyVarsR'] = keyVarsR
        params['base'] = base
    if 'globalrec' in x:
        gr_vars, gr_comb = GR_combination(obj, i, index)
        params['gr_vars'] = gr_vars
        params['gr_comb'] = gr_comb

    return params


def transformations(x, index, df_transf, i, params):
    """
    Apply transformation techniques to the dataframe.
    :param x: combination of techniques.
    :param index: index of combination.
    :param df_transf: input dataframe.
    :param i: index of dataframe.
    :param params: best parameters to apply the transformations.
    :return: transformed dataframe.
    """
    if 'sup' in x:
        df_transf = Suppression.best_per_sup(df_transf, params['per'])
    if 'topbot' in x:
        df_transf = TopBot.best_outer_fence(df_transf, params['keyVarsT'], params['outer_fence'])
    if 'noise' in x:
        rel_error = Noise.addNoise(df_transf)
        if (len(rel_error) != 0) and (not rel_error.equals(df_transf)):
            df_transf = Noise.assign_best_ep(rel_error, df_transf)
            pd.to_pickle(rel_error, 'Remote_results/relative_error_' + str(i) + '_' + str(index) + '.pkl')
    if 'round' in x:
        df_transf = Rounding.best_base_round(df_transf, params['keyVarsR'], params['base'])
    if 'globalrec' in x:
        # gr_vars, gr_comb = GR_combination(df_transf, i, index)
        if(len(params['gr_vars']) != 0) and (len(params['gr_comb']) != 0):
            df_transf = GlobalRec.best_bin_size(df_transf, params['gr_vars'], params['gr_comb'])

    return df_transf


# %%
@ray.remote
def process_single_df(df, i):
    """
    Process a dataframe and store the results.
    :param df: input dataframe.
    :param i: index of dataframe.
    :return: list of transformed dataframes and corresponding re-identification risk
                and combination of techniques applied.
    """
    print("Dataset #" + str(i))
    # list to store the transformed dataframes
    transformations_combs = []
    # dataframe to store re-identification risk
    reID_risk = pd.DataFrame(columns=['initial_fk'])
    df = change_cols_types(df)
    # keep target variable aside
    tgt = df.iloc[:, -1]
    # dataframe without target variable to apply transformation techniques
    df_val = df[df.columns[:-1]].copy()
    # create combinations adequate to the dataframe
    comb = define_combs(df_val)
    del comb[0]
    fk_var = 'fk_per'
    rl_var = 'rl_per'
    # calculate initial re-identification risk with k-anonymity
    reID_risk.loc[-1, 'initial_fk'] = CalcRisk.calc_max_fk(df_val)
    drop_comb = []
    drop_index = []
    drop_index_risk = []
    params = {}
    index = 0
    while index < len(comb):
        df_transf = df_val.copy()
        if len(comb[index]) < 2:
            # get best parameters
            params = parameters(comb[index], df_transf, df_val, i, params, index)
            # apply transformations
            df_transf = transformations(comb[index], index, df_transf, i, params)
            # store index of resulted transformations that have not suffer any change
            if df_transf.equals(df_val):
                drop_comb.append(comb[index])
                drop_index.append(index)
                drop_index_risk.append(index + 1)

        if (len(drop_comb) != 0) and (len(comb[index]) > 1):
            # remove combinations that does not happen due to defined criteria
            single_tuples = [item for item in comb if len(item) == 1]
            single_tuples = [item for item in single_tuples if item not in drop_comb]
            single_tuples = [item[0] for item in single_tuples]
            drop_tuples = []
            for i, item in enumerate(comb):
                for j in range(len(item)):
                    if item[j] not in single_tuples:
                        drop_tuples.append(i)

            comb = [comb[i] for i, _ in enumerate(comb) if i not in drop_tuples]
            reID_risk = reID_risk.drop(reID_risk.index[drop_index_risk]).reset_index(drop=True)
            reID_risk.index = np.arange(-1, len(reID_risk) - 1)
            transformations_combs = [transformations_combs[i] for i, _ in enumerate(transformations_combs) if
                                     i not in drop_index]
            index -= len(drop_comb)
            # reset lists
            drop_comb = []
            drop_index_risk = []

        if (len(comb) > 1) and (len(comb) != index):
            # params = parameters(comb[index], df_transf, df_val, i, params, index)
            # apply transformations after updating combinations
            if (len(drop_comb) == 0) and (len(comb[index]) > 1):
                df_transf = transformations(comb[index], index, df_transf, i, params)

        # recalculate re-identification risk with k-anonymity
        reID_risk.loc[index, fk_var] = CalcRisk.calc_max_fk(df_transf)
        # limit record linkage with blocking
        max_unique = max(df_transf.nunique())
        # avoid too many candidates to record linkage
        max_unique1 = set(df_transf.nunique())
        max_unique1 = [x for x in max_unique1]
        if (max_unique >= len(df_transf) * 0.85) and (len(max_unique1) != 1):
            max_unique1 = sorted(max_unique1)
            idx = [i for i, v in enumerate(max_unique1) if v < len(df_transf) * 0.85]
            if len(idx) != 0:
                block_column = df_transf.columns[df_transf.nunique() == max_unique1[idx[-1]]]
            else:
                block_column = df_transf.columns[df_transf.nunique() == max_unique1[0]]
        else:
            block_column = df_transf.columns[df_transf.nunique() == max_unique]

        if "sup" in comb[index]:
            # make sure that both datasets has equal dtypes to all columns for record linkage work
            check_types = df_val.dtypes.eq('object') == df_transf.dtypes.eq('object')
            idx = np.where(check_types == False)[0]
            if len(idx) >= 1:
                cols = df_val.columns[idx]
                for col in cols:
                    df_val[col] = df_val[col].astype(str)
        # calculate re-identification risk with record linkage
        try:
            reID_risk.loc[index, rl_var] = CalcRisk.calc_max_rl(df_transf, df_val, block_column[0],
                                                                indexer="block")
        except:
            warnings.warn("ERROR with record linkage on dataframe #" + str(i))
            reID_risk.loc[index, rl_var] = np.nan
        # reset df_val because of type change due to the suppression
        df_val = df[df.columns[:-1]]
        # add target to the transformed dataset
        df_transf[tgt.name] = tgt.values
        # store all transformed results
        transformations_combs.append(df_transf)

        gc.collect()
        print('Tech combs: ' + str(index) + '/' + str(len(comb)))
        index += 1

    return transformations_combs, reID_risk, comb


# %%
ds = load_data()  # 74 datasets
ds1 = ds[0:33].copy()
ds2 = ds[33:45].copy()
ds3 = ds[45:58].copy()
ds5 = ds[63:].copy()

# start ray
ray.init()

result_ids = []
c = 58
for i in range(0, len(ds5)):
    result_ids.append(process_single_df.remote(ds5[i], c))
    c += 1

# get ray results
all_risk = []
all_combs = []
all_transf_combs = []

for i in range(len(ds3)):
    # aggregate all of the results
    transf_combs, risk, combs = ray.get(result_ids[i])
    all_transf_combs.append(transf_combs)
    all_risk.append(risk)
    all_combs.append(combs)

# save ray results
pd.to_pickle(all_transf_combs, 'Remote_results/all_transf_combs_3.pkl')
pd.to_pickle(all_risk, 'Remote_results/all_risk_3.pkl')
pd.to_pickle(all_combs, 'Remote_results/all_combs_3.pkl')

# close Ray
ray.shutdown()

# %% Run one by one in DS4
ds4 = ds[58:63].copy()
c = 58
for i in range(len(list)):
    tr, r, cb = process_single_df(ds4[i].copy(), c)
    pd.to_pickle(tr, 'Remote_results/transf_combs_' + str(c) + '.pkl')
    pd.to_pickle(r, 'Remote_results/risk_' + str(c) + '.pkl')
    pd.to_pickle(cb, 'Remote_results/combs_' + str(c) + '.pkl')
    c += 1


def separeted_results():
    final_transf_combs = []
    final_risk = []
    final_combs = []
    for j in range(58, 63):
        all_transf = pd.read_pickle('Remote_results/transf_combs_' + str(j) + '.pkl')
        all_risk = pd.read_pickle('Remote_results/risk_' + str(j) + '.pkl')
        all_combs = pd.read_pickle('Remote_results/combs_' + str(j) + '.pkl')
        # if (j==58) or (j==59):
            # all_transf = list(itertools.chain(*all_transf))
            # all_risk = all_risk[0]
            # all_combs = list(itertools.chain(*all_combs))
        final_transf_combs.append(all_transf)
        final_risk.append(all_risk)
        final_combs.append(all_combs)
    # final_transf_combs = list(itertools.chain(*final_transf_combs))
    # final_risk = list(itertools.chain(*final_risk))
    # final_combs = list(itertools.chain(*final_combs))

    pd.to_pickle(final_transf_combs, 'Remote_results/all_transf_combs_4.pkl')
    pd.to_pickle(final_risk, 'Remote_results/all_risk_4.pkl')
    pd.to_pickle(final_combs, 'Remote_results/all_combs_4.pkl')


separeted_results()


# %%
def join_all_results():
    final_transf_combs = []
    final_risk = []
    final_combs = []
    for i in range(1, 6):
        all_transf_combs = pd.read_pickle('Remote_results/all_transf_combs_' + str(i) + '.pkl')
        all_risk = pd.read_pickle('Remote_results/all_risk_' + str(i) + '.pkl')
        all_combs = pd.read_pickle('Remote_results/all_combs_' + str(i) + '.pkl')
        final_transf_combs.append(all_transf_combs)
        final_risk.append(all_risk)
        final_combs.append(all_combs)

    final_transf_combs = list(itertools.chain(*final_transf_combs))
    final_risk = list(itertools.chain(*final_risk))
    final_combs = list(itertools.chain(*final_combs))

    pd.to_pickle(final_transf_combs, 'Final_results/final_transf_combs5.pkl')
    pd.to_pickle(final_risk, 'Final_results/final_risk5.pkl')
    pd.to_pickle(final_combs, 'Final_results/final_combs5.pkl')


join_all_results()
