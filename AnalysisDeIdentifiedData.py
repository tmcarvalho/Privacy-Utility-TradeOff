import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from venn import venn
from matplotlib.colors import LinearSegmentedColormap

# %% load data
ds = np.load('DS_clean.npy', allow_pickle=True)

transf = pd.read_pickle('Final_results/final_transf_combs2.pkl')
risk = pd.read_pickle('Final_results/final_risk2.pkl')
combs = pd.read_pickle('Final_results/final_combs2.pkl')


# %% deal with duplicates after all transformations
def repeated_dfs(i):
    drop_df = []
    for index, df in enumerate(transf[i]):
        # check if a transformation is empty
        if len(transf[i][index]) == 0:
            print(i)
        for idx, df1 in enumerate(transf[i][index + 1:]):
            if df1.equals(df):
                drop_df.append(index + idx + 1)
        drop_df = list(set(drop_df))

    return drop_df


def update_results(combs, risk, transf, drop_df):
    comb = [combs[j] for j, _ in enumerate(combs) if j not in drop_df]
    drop_index_risk = [x + 1 for x in drop_df]
    risk = risk.drop(risk.index[drop_index_risk]).reset_index(drop=True)
    risk.index = np.arange(-1, len(risk) - 1)
    transf = [transf[j] for j, _ in enumerate(transf) if j not in drop_df]

    return comb, risk, transf


indexes = pd.DataFrame(columns=['df'])
for i in range(len(transf)):
    drop_df = repeated_dfs(i)
    if len(drop_df) != 0:
        indexes.loc[i, 'df'] = drop_df
        # remove all duplicated transformed datasets
        combs[i], risk[i], transf[i] = update_results(combs[i], risk[i], transf[i], drop_df)

# %% get the total number of transformed dataset
size = 0
max_comb = 0
for i in range(len(transf)):
    if len(transf[i]) == 0:  # 0 datasets
        print(i)
    size += len(transf[i])  # 462 transformed datasets
    if len(transf[i]) == 31:  # 1 dataset
        max_comb += 1


# %% binary heatmap
heatmap = pd.DataFrame(columns=['Sup', 'Noise', 'Round', 'TopBot', 'GlobalRec', 'ds'])

for i in range(len(combs)):
    single_tuples = [item for item in combs[i] if len(item) == 1]
    sup = [item for item in single_tuples if 'sup' in item]
    noi = [item for item in single_tuples if 'noise' in item]
    round = [item for item in single_tuples if 'round' in item]
    topbot = [item for item in single_tuples if 'topbot' in item]
    globalrec = [item for item in single_tuples if 'globalrec' in item]
    if len(sup) != 0:
        heatmap.loc[i, 'Sup'] = 1
        heatmap.loc[i, 'ds'] = 'ds' + str(i)
    if len(noi) != 0:
        heatmap.loc[i, 'Noise'] = 1
        heatmap.loc[i, 'ds'] = 'ds' + str(i)
    if len(round) != 0:
        heatmap.loc[i, 'Round'] = 1
        heatmap.loc[i, 'ds'] = 'ds' + str(i)
    if len(topbot) != 0:
        heatmap.loc[i, 'TopBot'] = 1
        heatmap.loc[i, 'ds'] = 'ds' + str(i)
    if len(globalrec) != 0:
        heatmap.loc[i, 'GlobalRec'] = 1
        heatmap.loc[i, 'ds'] = 'ds' + str(i)

heatmap = heatmap.fillna(0)
heatmap = heatmap.set_index('ds')

heatmap.columns = ['Suppression', 'Noise', 'Round', 'Top&Bottom' + '\n' + 'Coding', 'Global' + '\n' + 'Recoding']

colors = ["lightgray", "gray"]
cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
ax = sns.heatmap(heatmap, cmap=cmap)
ax.set_title("Transformations techniques in each dataset")
ax.set_ylabel("")
# Set the colorbar labels
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['0', '1'])
# fig = ax.get_figure()
# fig.savefig("TransfTech_v2.pdf")

plt.show()


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
for i in range(len(comb)):
    comb[i] = [xs + (str(i),) for xs in comb[i]]
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

ax = venn(comb_dict, legend_loc="best", ax=None)
plt.show()
# fig = ax.get_figure()
# fig.savefig("venn.pdf")

df = ds[43].copy()
combs[43]
dff = ds[52].copy()
dfff = ds[54].copy()
df1 = transf[43][0]


