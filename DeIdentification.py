from transformations import TopBot
import numpy as np

# %% Load data
ds = np.load('DS_clean.npy', allow_pickle=True)

df = ds[1]

# %% Tob&Bottom Coding
x = TopBot.topBottomCoding(obj=df, value=1.6, replacement=1.6, kind='top', column='oldpeak')
