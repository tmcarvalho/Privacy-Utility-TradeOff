from transformations import TopBot, GlobalRec, Suppression, RoundFloats, Noise
from ReIdentificationRisk import kAnon, RecordLinkage
import DeIdentification
import itertools
import pandas as pd
import time
import random
import warnings
import numpy as np

# %%
ds = DeIdentification.load_data()
ds = random.choices(ds, k=10)

# df = ds[12].copy()

transf_tech = ['topbot', 'globalrec', 'sup', 'round', 'noise']
comb = [c for i in range(len(transf_tech) + 1) for c in itertools.combinations(transf_tech, i)]

reID_risk = pd.DataFrame(columns=['initial_fk'])


def transformations(x, df_transf):
    if 'topbot' in x:
        df_transf = TopBot.topBottomCoding(df_transf)
    if 'sup' in x:
        df_transf = Suppression.suppression(df_transf)
    if 'round' in x:
        df_transf = RoundFloats.roundFloats(df_transf)
    if 'noise' in x:
        df_transf = Noise.addNoise(df_transf)
    if 'globalrec' in x:
        df_transf = GlobalRec.globalRecoding(df_transf)
    return df_transf


start_time = time.time()

for i in range(0, len(ds)):
    ds[i] = DeIdentification.change_cols_types(ds[i])
    df_val = ds[i][ds[i].columns[:-1]]
    fk_var = 'fk_per_' + str(i)
    rl_var = 'rl_per_' + str(i)
    for index, x in enumerate(comb):
        df_transf = df_val.copy()
        if index == 0:
            reID_risk.loc[i, 'initial_fk'] = kAnon.calc_max_risk(df_transf)
        df_transf = transformations(x, df_transf)
        max_unique = max(df_transf.nunique())
        block_column = df_transf.columns[df_transf.nunique() == max_unique]
        reID_risk.loc[index, fk_var] = kAnon.calc_max_risk(df_transf)
        reID_risk.loc[index, rl_var] = RecordLinkage.calcRL(df_transf, df_val, block_column[0])
        print('Tech combs: ' + str(index) + '/' + str(len(comb)))

    if reID_risk[rl_var].min() == 100:
        warnings.warn("Dataframe is at max risk by record linkage!")
    elif reID_risk[rl_var].min() == 0:
        warnings.warn("Dataframe does not have observations with max risk by record linkage!")
    idx_min = np.argmin(reID_risk[rl_var].values)
    comb_idx = comb[idx_min]
    ds[i] = transformations(comb_idx, df_val)
    print(i)

end_time = time.time()

total_time = end_time-start_time

df_orig = ds[i][ds[i].columns[:-1]]

y = transformations(x, df_val)