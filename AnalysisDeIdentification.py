import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gs
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from venn import venn

# %% load actual data
ds = np.load('Data/DS_clean2.npy', allow_pickle=True)
transf = pd.read_pickle('Data/Final_results/final_transf_combs.pkl')
risk = pd.read_pickle('Data/Final_results/final_risk.pkl')
combs = pd.read_pickle('Data/Final_results/final_combs.pkl')


# %% get the total number of transformed dataset
def sizes(transf, comb):
    size = 0
    size_c = 0
    max_comb = 0
    for i in range(len(transf)):
        size += len(transf[i])  # 1146 transformed datasets
        size_c += len(comb[i])
        if len(transf[i]) == 31:  # 19 dataset
            max_comb += 1
            print(i)
    return size, size_c, max_comb


size, size_c, max_comb = sizes(transf, combs)


# %% heatmap with re-identification risk
# sort tuples
for idx in range(len(combs)):
    for i in range(len(combs[idx])):
        combs[idx][i] = tuple(sorted(combs[idx][i]))

# assign transformations with initials
cb = combs.copy()
for i in range(len(cb)):
    cb[i] = [tuple(s if s != "round" else "R" for s in tup) for tup in cb[i]]
    cb[i] = [tuple(s if s != "globalrec" else "G" for s in tup) for tup in cb[i]]
    cb[i] = [tuple(s if s != "noise" else "N" for s in tup) for tup in cb[i]]
    cb[i] = [tuple(s if s != "sup" else "S" for s in tup) for tup in cb[i]]
    cb[i] = [tuple(s if s != "topbot" else "T" for s in tup) for tup in cb[i]]

# create new dataframe with risk and percentage difference between the original and transformed datasets
reID = pd.DataFrame(columns=['ds', 'comb', 'initial_fk', 'fk', 'rl', 'rank', 'diff'])
for idx in range(len(transf)):
    for i, c in enumerate(cb[idx]):
        index = i
        if idx != 0:
            index = len(reID) + 1
        s = ", ".join([str(s) for s in list(c)])
        s = s.replace("'", "")
        reID.loc[index, 'comb'] = s
        reID.loc[index, 'ds'] = str(idx)
        reID.loc[index, 'initial_fk'] = risk[idx].loc[-1, 'initial_fk']
        reID.loc[index, 'fk'] = risk[idx].loc[i, 'fk_per']
        reID.loc[index, 'rl'] = risk[idx].loc[i, 'rl_per']
        reID.loc[index, 'rank'] = risk[idx].loc[i, 'rank']
        # 100 * (Sc - Sb) / Sb
        reID.loc[index, 'diff'] = (100 * abs(risk[idx].loc[i, 'rl_per'] - risk[idx].loc[-1, 'initial_fk'])) / \
                                  risk[idx].loc[-1, 'initial_fk']

# reID.to_csv("tilemap.csv", sep='\t', index=False)

# add values of original dataset
reID_org = pd.DataFrame(columns=['comb', 'initial_fk'])
for idx in range(len(transf)):
    index = len(reID_org) + 1
    reID_org.loc[index, 'comb'] = 'O'
    reID_org.loc[index, 'ds'] = str(idx)
    reID_org.loc[index, 'initial_fk'] = risk[idx].loc[-1, 'initial_fk']

reID_org['ds'] = reID_org['ds'].astype(np.int)
reID_org = reID_org.pivot_table(values='initial_fk', index=reID_org.ds, columns='comb', aggfunc='first')
reID_org['O'] = reID_org['O'].astype(int)

reID_red = reID[['ds', 'comb', 'diff']]
reID_red['ds'] = reID_red['ds'].astype(np.int)
reID_red['diff'] = reID_red['diff'].astype(np.int)
reID_red = reID_red.pivot_table(values='diff', index=reID_red.ds, columns='comb', aggfunc='first')
reID_red.fillna(value=np.nan, inplace=True)
col = reID_red.columns.tolist()
col = sorted(col, key=len)
reID_red = reID_red.reindex(col, axis=1)

# initiate tilemap
fig = plt.figure(figsize=(25, 35))
fig.subplots_adjust(wspace=0.01)
_, cols_org = reID_org.shape
_, cols_transf = reID_red.shape
grid = gs.GridSpec(nrows=1, ncols=2, height_ratios=[60], width_ratios=[cols_org, cols_transf])
ax1 = fig.add_subplot(grid[0])
ax2 = fig.add_subplot(grid[1], sharey=ax1)
# cax = fig.add_subplot(grid[:, 1])
cmap1 = sns.cubehelix_palette(rot=-.4, dark=0.3, light=0.6, as_cmap=True, reverse=True)
cmap2 = sns.cubehelix_palette(dark=0.3, light=0.6, as_cmap=True)
sns.heatmap(reID_org, cmap=cmap2, ax=ax1, cbar=False, annot=True, fmt='', linewidths=1, annot_kws={"size": 18})
# fig.colorbar(ax1.collections[0], ax=ax1, location="left", use_gridspec=False)
ax1.yaxis.set_ticks_position('left')
# cbar_kws={"shrink": .4},
sns.heatmap(reID_red, cmap=cmap1, ax=ax2, cbar=False, annot=True, fmt='g', linewidths=1, annot_kws={"size": 18})
# fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
ax_divider1 = make_axes_locatable(ax1)
cax1 = ax_divider1.append_axes('top', size='1.5%', pad='1%')
colorbar(ax1.get_children()[0], cax=cax1, orientation='horizontal')
cax1.xaxis.set_ticks_position('top')
cax1.tick_params(labelsize=15)
ax_divider = make_axes_locatable(ax2)
cax = ax_divider.append_axes('top', size='1.5%', pad='1%')
colorbar(ax2.get_children()[0], cax=cax, orientation='horizontal')
cax.xaxis.set_ticks_position('top')
cax.tick_params(labelsize=15)
# colorbar.set_label("'Percentage difference of re-identification risk'")
# ax1.yaxis.tick_right()
ax2.yaxis.tick_right()
plt.setp(ax1.get_xticklabels(), rotation=45)
plt.setp(ax1.get_yticklabels(), rotation=0)
plt.setp(ax2.get_xticklabels(), rotation=45)
plt.setp(ax2.get_yticklabels(), visible=False)
# ax2.set_xticks(np.arange(len(reID_red.columns)))
ax2.set(xlabel=None, ylabel=None)
ax1.set(xlabel=None)
ax1.set_ylabel(ylabel='Dataset Nr.', fontsize=26)
# ax1.set_yticklabels(ax1.get_yticks(), size=10)
ax1.tick_params(labelsize=22)
ax2.tick_params(labelsize=22, right=False)
sns.set_style("dark")
plt.show()
# fig.savefig("Plots/heatmap.pdf", dpi=300, bbox_inches='tight')


# %% Venn diagram
def assign_comb(comb_dict, item):
    elems = [i for i in item]
    el = ', '.join(str(element) for element in elems)
    for e in elems:
        if e == 'sup':
            comb_dict['Suppression'].add(el)
        if e == 'globalrec':
            comb_dict['GlobalRec'].add(el)
        if e == 'topbot':
            comb_dict['Top&Bottom'].add(el)
        if e == 'noise':
            comb_dict['Noise'].add(el)
        if e == 'round':
            comb_dict['Round'].add(el)

    return comb_dict


comb_dict = {'Suppression': set(), 'Noise': set(), 'Round': set(), 'Top&Bottom': set(), 'GlobalRec': set()}
comb = combs.copy()
for i in range(0, len(comb)):
    comb[i] = [xs + (str(i),) for xs in comb[i]]
    # comb[i] = ['_'.join(words) for words in comb[i]]
    for item in comb[i]:
        if 'sup' in item:
            if len(item) == 1:
                comb_dict['Suppression'].add(item[0])
            else:
                comb_dict = assign_comb(comb_dict, item)
        if 'noise' in item:
            if len(item) == 1:
                comb_dict['Noise'].add(item[0])
            else:
                comb_dict = assign_comb(comb_dict, item)
        if 'round' in item:
            if len(item) == 1:
                comb_dict['Round'].add(item[0])
            else:
                comb_dict = assign_comb(comb_dict, item)
        if 'topbot' in item:
            if len(item) == 1:
                comb_dict['Top&Bottom'].add(item[0])
            else:
                comb_dict = assign_comb(comb_dict, item)
        if 'globalrec' in item:
            if len(item) == 1:
                comb_dict['GlobalRec'].add(item[0])
            else:
                comb_dict = assign_comb(comb_dict, item)

names = ['Suppression', 'Noise', 'Rounding', 'Top&Bottom', 'Globar Recoding']
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()
venn(comb_dict, legend_loc="best", ax=ax)
plt.show()
fig = ax.get_figure()
# fig.savefig("Plots/venn.pdf", dpi=300, bbox_inches='tight')

