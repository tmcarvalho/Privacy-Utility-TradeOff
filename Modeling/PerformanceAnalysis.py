# %%
import re
from os import walk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% load data
ds = np.load('Data/DS_clean2.npy', allow_pickle=True)
combs = pd.read_pickle('Data/Final_results/final_combs.pkl')

transf_folder = 'Data/Final_results/AlloutputFiles/Transformed'
_, transf_folders, _ = next(walk(transf_folder))

org_folder = 'Data/Final_results/AlloutputFiles/Originals'
_, org_folders, _ = next(walk(org_folder))

# %%
# store the position of the datasets that have the 5 solutions at the same time
lst_all_solutions = []
for idx, comb in enumerate(combs):
    if len(comb) == 31:
        lst_all_solutions.append(idx)  # 18 datasets with all solutions


# %%
def join_res(flag, master_folder, folder):
    '''
    Concatenate each algorithm results
    :param flag: 0 - transformed folder; 1- original folder
    :param master_folder: initial path
    :param folder: each folder from master_folder
    :return: dataframe with concatenated results and specified model
    '''
    rf_lst = bag_lst = xgb_lst = logr_lst = nn_lst = []
    r = b = x = l = n = 0
    _, _, files = next(walk(f'{master_folder}/{folder}'))
    for file in files:
        if (flag != 0) and ('csv' in file) and ('_diff' not in file):
            if 'rf' in file:
                rf = pd.read_csv(f'{master_folder}/{folder}/{file}', sep='\t')
                rf['model'] = "RF"
                # if len(rf[rf['mean_test_f1_perdif'] >= 100]):
                #    lst_file.append(file)
                if r == 0:
                    rf_lst = rf
                    r += 1
                else:
                    rf_lst = pd.concat([rf_lst, rf])
            if 'bag' in file:
                bag = pd.read_csv(f'{master_folder}/{folder}/{file}', sep='\t')
                bag['model'] = "Bag"
                if b == 0:
                    bag_lst = bag
                    b += 1
                else:
                    bag_lst = pd.concat([bag_lst, bag])
            if 'xgb' in file:
                xgb = pd.read_csv(f'{master_folder}/{folder}/{file}', sep='\t')
                xgb['model'] = "XGB"
                if x == 0:
                    xgb_lst = xgb
                    x += 1
                else:
                    xgb_lst = pd.concat([xgb_lst, xgb])
            if 'logr' in file:
                logr = pd.read_csv(f'{master_folder}/{folder}/{file}', sep='\t')
                logr['model'] = "LogR"
                if l == 0:
                    logr_lst = logr
                    l += 1
                else:
                    logr_lst = pd.concat([logr_lst, logr])
            if 'nn' in file:
                nn = pd.read_csv(f'{master_folder}/{folder}/{file}', sep='\t')
                nn['model'] = "NN"
                if n == 0:
                    nn_lst = nn
                    n += 1
                else:
                    nn_lst = pd.concat([nn_lst, nn])

        else:
            pass

    if len(rf_lst) != 0:
        res = pd.concat([rf_lst, bag_lst, xgb_lst, logr_lst, nn_lst])
        return res


# %%
# Each solution in ONE plot
def each_transf(flag, transf, master_folder, folders):
    """
    Concatenate all results and assign dataset name
    :param flag: 0 - transformed folder; 1- original folder
    :param transf: list with original and transformed indexes
    :param master_folder: folder that indicates the initial path
    :param folders: each folder from master_folder
    :return: dataframe with concatenated results and dataset name
    """
    sol = pd.DataFrame()
    for folder in folders:
        if flag == 0:
            for t in transf:
                if f'ds{t[0]}_transf{t[1]}' == folder:
                    solution = join_res(1, master_folder, folder)
                    solution['ds'] = f'ds{t[0]}_transf{t[1]}'
                    sol = pd.concat([sol, solution])
        else:
            solution = join_res(1, master_folder, folder)
            nr = int(int(re.search(r'\d+', folder)[0]))
            solution['ds'] = f'ds{nr}'
            sol = pd.concat([sol, solution])

    return sol


cb = combs.copy()
for i in range(len(cb)):
    cb[i] = [tuple(s if s != "round" else "R" for s in tup) for tup in cb[i]]
    cb[i] = [tuple(s if s != "globalrec" else "G" for s in tup) for tup in cb[i]]
    cb[i] = [tuple(s if s != "noise" else "N" for s in tup) for tup in cb[i]]
    cb[i] = [tuple(s if s != "sup" else "S" for s in tup) for tup in cb[i]]
    cb[i] = [tuple(s if s != "topbot" else "T" for s in tup) for tup in cb[i]]

transfs = ['G', 'N', 'R', 'S', 'T', 'G, N', 'G, R', 'G, S',
           'G, T', 'N, R', 'N, S', 'N, T', 'R, S', 'R, T',
           'S, T', 'G, N, R', 'G, N, S', 'G, N, T',
           'G, R, S', 'G, R, T', 'G, S, T', 'N, R, S',
           'N, R, T', 'N, S, T', 'R, S, T', 'G, N, R, S',
           'G, N, R, T', 'G, N, S, T', 'G, R, S, T',
           'N, R, S, T', 'G, N, R, S, T']


def lst_ds_(transf_name, lst_all_solutions):
    """
    List with indices of original and transformed dataset
    :param transf_name: transformation (solution) name
    :param lst_all_solutions: list with indices of the 18 datasets that contains all solutions
    :return: list with original and transformed indexes
    """
    lst = []
    for idx, comb in enumerate(cb):
        for i, c in enumerate(cb[idx]):
            split_tr = transf_name.split(', ')
            if len(lst_all_solutions) == 0:
                if (len(c) == len(split_tr)) and (all(t in c for t in split_tr)):
                    lst.append([idx, i])
            else:
                if (len(c) == len(split_tr)) and (all(t in c for t in split_tr)) and (idx in lst_all_solutions):
                    lst.append([idx, i])
    # print(lst)  # [original, transf]
    return lst


def add_solution_name(flag, lst_sol, master_folder, folders):
    """
    Add solution name and concatenate all results
    :param flag: 0 - transformed folder; 1- original folder
    :param lst_sol: list with indexes of the 18 datasets that contains all solutions
    :param master_folder: folder that indicates the initial path
    :param folders: each folder from master_folder
    :return: concatenated dataframe with all ML algorithms and solutions
    """
    df = pd.DataFrame()
    if flag == 0:
        for t in transfs:
            lst = lst_ds_(t, lst_sol)
            tr_res = each_transf(0, lst, master_folder, folders)
            tr_res['solution'] = t
            df = pd.concat([df, tr_res])

    else:
        tr_res = each_transf(1, [], master_folder, folders)
        df = pd.concat([df, tr_res])

    return df


# transformed results
all_results = add_solution_name(0, [], transf_folder, transf_folders)
all_results = all_results.reset_index(drop=True)
for i in range(0, len(all_results)):
    # remove comma from transformation name
    all_results.loc[i, 'solution'] = all_results.loc[i, 'solution'].replace(', ', '')
# all_results.to_csv('Data/all_solutions.csv', sep='\t', index=False)

# baseline results
original_results = add_solution_name(1, [], org_folder, org_folders)
# original_results.to_csv('Data/baseline_results.csv', sep='\t', index=False)


# %%
def ploting(data, transf_name):
    all_solutions_melt = data.melt(id_vars=['model'], value_vars=['mean_test_bal_acc_perdif',
                                                                  'mean_test_gmean_perdif',
                                                                  'mean_test_f1_weighted_perdif'],
                                   var_name='metrics', value_name='values')

    sns.set_style("darkgrid")
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    ax = sns.boxplot(x='model', y='values', hue='metrics', data=all_solutions_melt, ax=axs)
    ax.set_yscale('symlog')
    plt.xlabel('')
    plt.ylabel('Percentage difference', fontsize=14)
    plt.title(transf_name, fontsize=18)
    legend_labels, _ = ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['Balanced Accuracy', 'G-mean', 'F1-weighted'],
              # bbox_to_anchor=(1.05, 1),
              loc='lower center',
              borderaxespad=-7,
              ncol=3,
              title='Measures')
    # plt.legend(title='Measures', loc='best', bbox_to_anchor=(1, 1), ncol=1)
    sns.set(font_scale=1.5)
    plt.show()
    figure = ax.get_figure()
    figure.savefig(f'Plots/{transf_name}.png', bbox_inches='tight')


# %% plot each solution
# all_results = pd.read_csv('Data/all_solutions.csv', sep='\t')
grp_ds = all_results.groupby('solution')
for grp_name, df_group in grp_ds:
    ploting(df_group, df_group['solution'].unique()[0])


# %% plot all solutions with just the 18 datasets
sol18 = add_solution_name(0, lst_all_solutions, transf_folder, transf_folders)
sol18 = sol18.reset_index(drop=True)
for i in range(0, len(sol18)):
    sol18.loc[i, 'solution'] = sol18.loc[i, 'solution'].replace(', ', '')

# sol18.to_csv('Data/sol18.csv', sep='\t', index=False)
# sol18 = pd.read_csv('Data/sol18.csv', sep='\t')

all_results['comparison'] = 'All datasets'
sol18['comparison'] = '18 datasets'

two_comparisons = pd.concat([all_results, sol18])

two_comparisons_melt = two_comparisons.melt(id_vars=['solution', 'model', 'comparison'],
                                              value_vars=['mean_test_f1_weighted_perdif'],
                                              var_name='sol', value_name='values')

sns.set_style("darkgrid")
g = sns.catplot(x='values', y='solution', hue='comparison', col='model', data=two_comparisons_melt,
                kind='box', palette='colorblind',
                height=14, aspect=0.26, col_wrap=5, legend=False).set(xscale='symlog')
(g.set_axis_labels("Weighted Fscore", "")
 .set_titles("{col_name}")
 .set(xlim=(-10 ** 2, 10 ** 2))
 )
g.add_legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
# g.fig.suptitle('Percentage difference of weighted Fscore', fontsize=18)
# plt.legend(bbox_to_anchor=(0.5, -0.1), loc='lower center', ncol=2, borderaxespad=0.)
plt.tight_layout()
plt.show()
# plt.savefig(f'Plots/Percentage Difference of Fscore (all vs 18).pdf', bbox_inches='tight')

# %% table with rank - performance
all_results_max = two_comparisons.groupby(['ds', 'model', 'solution', 'comparison'])[
    'mean_test_f1_weighted'].max().reset_index()
all_results_max = two_comparisons.groupby(['model', 'solution', 'comparison'])[
    'mean_test_f1_weighted'].mean().reset_index()
all_results_max['rank'] = all_results_max.groupby(['model', 'comparison'])['mean_test_f1_weighted'].rank(
    ascending=False).astype(int)

all_results_max['model'] = np.where(all_results_max['model'] == 'Bag', 'Bagging', all_results_max['model'])
all_results_max['model'] = np.where(all_results_max['model'] == 'RF', 'Random Forest', all_results_max['model'])
all_results_max['model'] = np.where(all_results_max['model'] == 'NN', 'Neural Network', all_results_max['model'])
all_results_max['model'] = np.where(all_results_max['model'] == 'XGB', 'XGBoost', all_results_max['model'])
all_results_max['model'] = np.where(all_results_max['model'] == 'LogR', 'Logistic Regression', all_results_max['model'])


def custom_key(str):
    """
    Order string by length
    :param str: string
    :return: ordered string
    """
    return len(str), str.lower()


# get unique ordered comb values
row = all_results_max.solution.values.tolist()
row = sorted(set(row), key=custom_key)


def facet_heatmap(data, color, **kws):
    data = data.pivot_table(index=['comparison'], columns='solution', values='rank')
    data = data.reindex(row, axis=1)
    # pass kwargs to heatmap
    sns.heatmap(data, square=True, **kws)


sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(20, 10))
g = sns.FacetGrid(all_results_max, row="model", aspect=8.5)

cbar_ax = g.fig.add_axes([0.3, 0.05, .4, .02])
g = g.map_dataframe(facet_heatmap, annot=True, annot_kws={"size": 20}, cmap='YlGnBu', cbar_ax=cbar_ax, cbar_kws=dict(
    pad=0.1, shrink=0.3, label='Rank of performance', orientation='horizontal'))
g.set_titles("{row_name}")
# g.fig.subplots_adjust(bottom=.2)
g.set_xticklabels(rotation=30)
sns.set(font_scale=2.2)
# plt.tight_layout()
plt.show()
# plt.savefig(f'Plots/Performance rank.svg', bbox_inches='tight')
