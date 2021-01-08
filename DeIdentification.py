from transformations import TopBot, GlobalRec, Suppression, RoundFloats, Noise
from ReIdentificationRisk import kAnon, RecordLinkage
import numpy as np
import itertools
from decimal import Decimal
import pandas as pd
import time


# %% Load data
def load_data():
    dss = np.load('DS_clean.npy', allow_pickle=True)
    ds = dss.copy()
    return ds


# %%
def change_cols_types(df):
    """
    Remove trailing zeros and assign as 'int' type
    :param df: input data
    :return: transformed data
    """
    cols = df.select_dtypes(include=np.number).columns.values
    for col in cols:
        df[col] = df[col].apply(Decimal).astype(str)
        if any('.' in s for s in df[col]):
            df[col] = df[col].astype('float')
        else:
            df[col] = df[col].astype('int')
    return df


# %%
def deIdentification():
    ds = load_data()
    reID_risk = pd.DataFrame(columns=['initial_fk'])

    transf_tech = ['topbot', 'globalrec', 'sup', 'round', 'noise']

    comb = [c for i in range(len(transf_tech)+1) for c in itertools.combinations(transf_tech, i)]

    start_time = time.time()
    for i in range(0, len(ds)):
        ds[i] = change_cols_types(ds[i])
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

            reID_risk.loc[index, 'fk_comb_' + str(index)] = kAnon.calc_max_risk(df_transf)
            reID_risk.loc[index, 'rl_comb_' + str(index)] = RecordLinkage.recordLinkage(df_transf, df_val)
            print('Tech combs: ' + str(index) + '/' + str(len(comb)))
    end_time = time.time()


