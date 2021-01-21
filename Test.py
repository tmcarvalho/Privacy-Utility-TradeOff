import itertools
import psutil
from ReIdentificationRisk import CalcRisk
import DeIdentification
import pandas as pd
import time
import numpy as np
import ray
from itertools import chain
import pickle

# %%
ds = DeIdentification.load_data()  # 90 datasets
# ds = random.choices(ds, k=3)

df = ds[65].copy()

dfs = []
dfs.append(ds[7])
# dfs = ds[0:8]
df = ds[7].copy()
# [0, 7, 9, 11, 15, 17, 21, 22, 23, 28, 29, 30, 31, 34, 35, 37, 38]

# test ranges
all_ranges = []
magnitude = [1, 2]
df = DeIdentification.change_cols_types(df)
vars = df.select_dtypes(include=float).columns
for j in range(0, 24):
    all_ranges.append(magnitude)
comb = [i for i in itertools.product(*all_ranges)]


@ray.remote
def process_single_df(df, i):
    print(i)
    # list to store the transformed dataframes
    transformations_combs = []
    # dataframe to store re-identification risk
    reID_risk = pd.DataFrame(columns=['initial_fk'])
    df = DeIdentification.change_cols_types(df)
    # keep target variable aside
    tgt = df.iloc[:, -1]
    # dataframe without target variable to apply transformation techniques
    df_val = df[df.columns[:-1]].copy()
    # create combinations adequate to the dataframe
    comb = np.array(DeIdentification.define_combs(df_val), dtype='object')
    fk_var = 'fk_per'
    rl_var = 'rl_per'
    c = 0
    for index, x in enumerate(comb):
        df_transf = df_val.copy()
        if index == 0:
            # calculate initial re-identification risk with k-anonymity
            reID_risk.loc[index, 'initial_fk'] = CalcRisk.calc_max_fk(df_transf)
        else:
            # apply transformations
            df_transf, c = DeIdentification.transformations(x, df_transf, c, i)
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
            # record linkage with full indexer in cases where only exist dtype floats
            # if all(df_transf.dtypes == np.float):
            #     reID_risk.loc[index, rl_var] = CalcRisk.calc_max_rl(df_transf, df_val, block="",
            #                                                         indexer="full")

            # else:
            # limit record linkage with blocking! Block column shoudn't be float dtype
            # max_unique = max(df_transf.select_dtypes(exclude=np.float).nunique())
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

    pd.to_pickle(transformations_combs, 'Results_remote/transformations_' + str(i) + '.pkl')
    pd.to_pickle(reID_risk, 'Results_remote/risk_' + str(i) + '.pkl')
    pd.to_pickle(comb, 'Results_remote/comb_' + str(i) + '.pkl')

    return transformations_combs, reID_risk, comb


# start Ray
# num_cpus = psutil.cpu_count(logical=False)
# ray.init(num_cpus=num_cpus)
ray.init()

start_time = time.time()
result_ids = []
for i in range(len(dfs)):
    result_ids.append(process_single_df.remote(dfs[i], i))
    # process_single_df.remote(dfs[i], i)

end_time = time.time()
total_time = end_time - start_time

x = pd.read_pickle('Results_remote/transformations_2.pkl')
y = pd.read_pickle('Results_remote/risk_2.pkl')
z = pd.read_pickle('Results_remote/relative_error_1.pkl')

all_risk = []
all_combs = []
all_transf_combs = []

for i in range(len(dfs)):
    # aggregate all of the results
    transf_combs, risk, combs = ray.get(result_ids[i])
    all_transf_combs.append(transf_combs)
    all_risk.append(risk)
    all_combs.append(combs)

pd.to_pickle(all_transf_combs, 'Final_results/all_transf_combs.pkl')
pd.to_pickle(all_risk, 'Final_results/all_risk.pkl')
pd.to_pickle(all_combs, 'Final_results/all_combs.pkl')

x = pd.read_pickle('Final_results/all_transf_combs.pkl')
all_x = []
all_x.append(x)

all_x = list(chain(*all_x))

# close Ray
ray.shutdown()

all_transf_combs[4][4]
all_combs[4]

# 7min -> 0
# 2min -> 1
