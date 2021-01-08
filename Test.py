from transformations import TopBot, GlobalRec, Suppression, RoundFloats, Noise
from ReIdentificationRisk import kAnon, RecordLinkage
import DeIdentification
import numpy as np
import itertools
import pandas as pd
import time
import random

# %%
ds = DeIdentification.load_data()
ds = random.choices(ds, k=10)

# df = ds[12].copy()

transf_tech = ['topbot', 'globalrec', 'sup', 'round', 'noise']
comb = [c for i in range(len(transf_tech) + 1) for c in itertools.combinations(transf_tech, i)]

reID_risk = pd.DataFrame(columns=['initial_fk'])

start_time=time.time()

for i in range(0, len(ds)):
    ds[i] = DeIdentification.change_cols_types(ds[i])
    df_val = ds[i][ds[i].columns[:-1]]
    for index, x in enumerate(comb):
        df_transf = df_val.copy()
        if index == 0:
            reID_risk.loc[i, 'initial_fk'] = kAnon.calc_max_risk(df_transf)
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

        reID_risk.loc[index, 'fk_comb_'+str(index)] = kAnon.calc_max_risk(df_transf)
        reID_risk.loc[index, 'rl_comb_'+str(index)] = RecordLinkage.recordLinkage(df_transf, df_val)
        print('Tech combs: ' + str(index) + '/' + str(len(comb)))
end_time = time.time()

