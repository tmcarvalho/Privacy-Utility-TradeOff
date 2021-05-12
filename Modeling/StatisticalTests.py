import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
all_results = pd.read_csv('Data/all_solutions.csv', sep='\t')


# %%
def BayesianSignTest(diffVector, rope_min, rope_max):
    # add the fake first observation in 0
    # diffVector = pd.concat([pd.Series([0]), pd.Series([all_results_max_diff])])

    # for the moment we implement the sign test. Signedrank will follows
    probLeft = np.mean(diffVector < rope_min)
    probRope = np.mean((diffVector > rope_min) & (diffVector < rope_max))
    probRight = np.mean(diffVector > rope_max)
    alpha = [probLeft, probRope, probRight]
    alpha = [a + 0.0001 for a in alpha]
    res = np.random.dirichlet(alpha, 30000).mean(axis=0)

    return res


# %%
all_results_max = all_results.groupby(['ds', 'model', 'solution'])['mean_test_f1_weighted_perdif'].max().reset_index()

max_diff_bag = all_results_max[all_results_max['model'] == 'Bag'].mean_test_f1_weighted_perdif.reset_index(drop=True)
max_diff_rf = all_results_max[all_results_max['model'] == 'RF'].mean_test_f1_weighted_perdif.reset_index(drop=True)
max_diff_xgb = all_results_max[all_results_max['model'] == 'XGB'].mean_test_f1_weighted_perdif.reset_index(drop=True)
max_diff_nn = all_results_max[all_results_max['model'] == 'NN'].mean_test_f1_weighted_perdif.reset_index(drop=True)
max_diff_logr = all_results_max[all_results_max['model'] == 'LogR'].mean_test_f1_weighted_perdif.reset_index(drop=True)

bst_bag = BayesianSignTest(max_diff_bag, -1, 1)
bst_rf = BayesianSignTest(max_diff_rf, -1, 1)
bst_xgb = BayesianSignTest(max_diff_xgb, -1, 1)
bst_nn = BayesianSignTest(max_diff_nn, -1, 1)
bst_logr = BayesianSignTest(max_diff_logr, -1, 1)

model_res = pd.DataFrame(columns=['Model', 'Result', 'Probability'])
model_res.loc[0] = ['Bag', 'Lose', bst_bag[0]]
model_res.loc[1] = ['RF', 'Lose', bst_rf[0]]
model_res.loc[2] = ['XGB', 'Lose', bst_xgb[0]]
model_res.loc[3] = ['NN', 'Lose', bst_nn[0]]
model_res.loc[4] = ['LogR', 'Lose', bst_logr[0]]
model_res.loc[5] = ['Bag', 'Draw', bst_bag[1]]
model_res.loc[6] = ['RF', 'Draw', bst_rf[1]]
model_res.loc[7] = ['XGB', 'Draw', bst_xgb[1]]
model_res.loc[8] = ['NN', 'Draw', bst_nn[1]]
model_res.loc[9] = ['LogR', 'Draw', bst_logr[1]]
model_res.loc[10] = ['Bag', 'Win', bst_bag[2]]
model_res.loc[11] = ['RF', 'Win', bst_rf[2]]
model_res.loc[12] = ['XGB', 'Win', bst_xgb[2]]
model_res.loc[13] = ['NN', 'Win', bst_nn[2]]
model_res.loc[14] = ['LogR', 'Win', bst_logr[2]]


# %% plot
def move_legend(g, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    g.legend(handles, labels, ncol=4, loc=new_loc, bbox_to_anchor=(0.5, -.35), frameon=False, **kws)


sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(7, 3.5))
g = sns.histplot(model_res, x='Model', hue='Result', weights='Probability', edgecolor='none',
                 multiple='stack', palette=['tab:blue', 'orange', 'tab:green'], shrink=0.8)
g.set_ylabel('Probability')
g.set_xlabel('')
# fix the legend
move_legend(g, "lower center")
plt.yticks(np.arange(0, 1.25, 0.25))
plt.axhline(y=0.5, color='lightgrey', linestyle='-', linewidth=0.8)
plt.tight_layout()
# plt.show()
plt.savefig(f'Plots/probability.pdf')
