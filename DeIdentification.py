import warnings
from transformations import TopBot, GlobalRec, Suppression, RoundFloats, Noise
from ReIdentificationRisk import CalcRisk
import numpy as np
import itertools
from decimal import Decimal
import pandas as pd
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


def transformations(x, df_transf, c, i):
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
        rel_error = Noise.addNoise(df_transf)
        df_transf = Noise.assign_best_ep(rel_error, df_transf)
        pd.to_pickle(rel_error, 'Results_remote/relative_error_' + str(i) + '.pkl')
    if 'globalrec' in x:
        c += 1
        keyVars = comb = []
        if c == 1:
            keyVars, comb = GlobalRec.globalRecoding(df_transf)
            pd.to_pickle(comb, 'Results_remote/GRcomb_' + str(i) + '.pkl')
            if len(comb) != 0:
                df_transf = GlobalRec.best_bin_size(df_transf, keyVars, comb)
        else:
            if len(comb) != 0:
                df_transf = GlobalRec.best_bin_size(df_transf, keyVars, comb)

    return df_transf, c


# %%
@ray.remote
def process_single_df(df, i):
    print("Dataset #" + str(i))
    # list to store the transformed dataframes
    transformations_combs = []
    # dataframe to store re-identification risk
    reID_risk = pd.DataFrame(columns=['initial_fk'])
    c = 0
    df = change_cols_types(df)
    # keep target variable aside
    tgt = df.iloc[:, -1]
    # dataframe without target variable to apply transformation techniques
    df_val = df[df.columns[:-1]].copy()
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
            df_transf, c = transformations(x, df_transf, c, i)
            if "sup" in x:
                # make sure that both datasets has equal dtypes to all columns
                check_types = df_val.dtypes.eq('object') == df_transf.dtypes.eq('object')
                idx = np.where(check_types == False)[0]
                if len(idx) >= 1:
                    cols = df_val.columns[idx]
                    df_val.loc[:, cols] = df_val.loc[:, cols].astype(str)
                else:
                    continue
            # recalculate re-identification risk with k-anonymity
            reID_risk.loc[index, fk_var] = CalcRisk.calc_max_fk(df_transf)
            # limit record linkage with blocking!
            max_unique = max(df_transf.nunique())
            block_column = df_transf.columns[df_transf.nunique() == max_unique]
            # calculate record linkage
            try:
                reID_risk.loc[index, rl_var] = CalcRisk.calc_max_rl(df_transf, df_val, block_column[0],
                                                                    indexer="block")
            except:
                warnings.warn("ERROR with record linkage on dataframe #" + str(i))
                reID_risk.loc[index, rl_var] = np.nan
            # reset df_val because of dtype change due to the suppression
            df_val = df[df.columns[:-1]]
            # add target to the transformed dataset
            df_transf[tgt.name] = tgt.values
            # save all results
            transformations_combs.append(df_transf)

        print('Tech combs: ' + str(index) + '/' + str(len(comb)))

    pd.to_pickle(transformations_combs, 'Results_remote/transformations_' + str(i) + '.pkl')
    pd.to_pickle(reID_risk, 'Results_remote/risk_' + str(i) + '.pkl')
    pd.to_pickle(comb, 'Results_remote/comb_' + str(i) + '.pkl')

    return transformations_combs, reID_risk, comb


# %%
ds = load_data()  # 87 datasets
ds1 = ds[0:45]
ds2 = ds[45:66]
ds3 = ds[66:]

# start Ray
ray.init()

result_ids = []
c = 45  # to apply in second part
for i in range(0, len(ds2)):
    result_ids.append(process_single_df.remote(ds2[i], c))
    c += 1

all_risk = []
all_combs = []
all_transf_combs = []

for i in range(len(ds2)):
    # aggregate all of the results
    transf_combs, risk, combs = ray.get(result_ids[i])
    all_transf_combs.append(transf_combs)
    all_risk.append(risk)
    all_combs.append(combs)


pd.to_pickle(all_transf_combs, 'Final_results/all_transf_combs_2.pkl')
pd.to_pickle(all_risk, 'Final_results/all_risk_2.pkl')
pd.to_pickle(all_combs, 'Final_results/all_combs_2.pkl')

x = pd.read_pickle('Final_results/all_transf_combs_1.pkl')
y = pd.read_pickle('Final_results/all_risk_1.pkl')
z = pd.read_pickle('Final_results/all_combs_1.pkl')

# close Ray
ray.shutdown()
