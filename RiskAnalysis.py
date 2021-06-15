# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# %%
combs = pd.read_pickle('Data/Final_results/final_combs.pkl')

# %%
# store the position of the datasets that have the 5 solutions at the same time
lst_all_solutions = []
for idx, comb in enumerate(combs):
    if len(comb) == 31:
        lst_all_solutions.append(idx)  # 18 datasets with all solutions

risk = pd.read_csv('Data/tilemap.csv', sep='\t')
risk['comparison'] = 'All datasets'

for i in range(0, len(risk)):
    if risk['ds'][i] in lst_all_solutions:
        risk = risk.append(risk.loc[i], ignore_index=True)
        risk.loc[len(risk) - 1, 'comparison'] = '18 datasets'

risk_mean = risk.groupby(['comb', 'comparison'])['rl'].mean().reset_index()
risk_mean['rank'] = risk_mean.groupby(['comparison'])["rl"].rank().astype(int)
risk_mean.describe()


def custom_key(str):
    return len(str), str.lower()


def order_combs(df):
    # get unique ordered comb values
    row = df.comb.values.tolist()
    row = sorted(set(row), key=custom_key)

    # apply order
    df = df.loc[pd.Series(pd.Categorical(df.comb, categories=row)) \
        .sort_values().index]
    return df


risk = order_combs(risk)
risk_mean = order_combs(risk_mean)

# Barplot with percentage difference of risk
sns.set_style("darkgrid")
fig = plt.figure(figsize=(12, 6))
sns.barplot(data=risk, x="comb", y="diff", hue="comparison", palette='colorblind')
sns.set_theme(style="whitegrid", font_scale=0.8)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Percentage difference")
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.18), borderaxespad=0.)
plt.show()
# plt.tight_layout()
# plt.savefig(f'Plots/bodega4.pdf', bbox_inches='tight')

# %% Table with rank
heatmap = pd.pivot_table(risk_mean, values='rank', index=['comb'], columns='comparison')
row = risk_mean.comb.values.tolist()
row = sorted(set(row), key=custom_key)
heatmap = heatmap.reindex(row)
heatmap.head(n=5)
heatmap = heatmap.T

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(16, 10))
ax = sns.heatmap(heatmap, cmap="YlGnBu", annot=True, annot_kws={"size": 12},
                 cbar_kws=dict(use_gridspec=False, location="bottom", pad=0.1, shrink=0.3,
                               label='Rank of re-identification risk'),
                 square=True)

# fix cbar ticks
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(axis='both', which='both', length=0)

plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=30)
sns.set(font_scale=1.1)
plt.show()
# plt.savefig(f'Plots/bodega50.pdf', bbox_inches='tight')

# %% maximum privacy and maximum performance with percentage difference
# select maximum risk for the 18 datasets
risk_max = risk[(risk['comparison'] == '18 datasets') & (risk['rl'] == 0)]
risk_max = risk_max[['ds', 'comb', 'rl']]

# load predictive performance results for the 18 datasets
sol_30 = pd.read_csv('Data/sol_30.csv', sep='\t')
sol_30 = sol_30[['model', 'mean_test_f1_weighted_perdif', 'ds', 'solution']]

# get number of ds
sol_30['ds'] = sol_30['ds'].apply(lambda x: int(re.search(r'\d+', x.split('_')[0])[0]))

# merge predictive performance and re-identification risk
max_performance_risk = pd.merge(sol_30, risk_max, how='left', left_on=['ds', 'solution'], right_on=['ds', 'comb'])

# clean nan
max_performance_risk = max_performance_risk[max_performance_risk['rl'].notna()]

# del unnecessary columns
del max_performance_risk['comb']
del max_performance_risk['rl']

max_performance_risk['grp'] = max_performance_risk.groupby(['model', 'ds', 'solution'])['mean_test_f1_weighted_perdif'].transform(lambda x: x.max() if x.max() > 0 else x.max())
# add missing values
max_performance_risk.loc[max_performance_risk['mean_test_f1_weighted_perdif'] != max_performance_risk['grp'], 'grp'] = np.nan
# remove missing values
max_performance_risk = max_performance_risk[max_performance_risk['grp'].notna()]
# remove grp
del max_performance_risk['grp']
# remove draws by
max_performance_risk = max_performance_risk.reset_index(drop=True)
max_performance_risk = max_performance_risk.loc[max_performance_risk.groupby(['model', 'ds', 'solution'])['mean_test_f1_weighted_perdif'].idxmax()]
max_performance_risk = max_performance_risk.reset_index(drop=True)

# boxplot for each model
max_performance_risk['model'] = np.where(max_performance_risk['model'] == 'Bag', 'Bagging', max_performance_risk['model'])
max_performance_risk['model'] = np.where(max_performance_risk['model'] == 'RF', 'Random Forest', max_performance_risk['model'])
max_performance_risk['model'] = np.where(max_performance_risk['model'] == 'NN', 'Neural Network', max_performance_risk['model'])
max_performance_risk['model'] = np.where(max_performance_risk['model'] == 'XGB', 'XGBoost', max_performance_risk['model'])
max_performance_risk['model'] = np.where(max_performance_risk['model'] == 'LogR', 'Logistic Regression', max_performance_risk['model'])

fig, ax = plt.subplots(figsize=(6, 5))
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="model", y="mean_test_f1_weighted_perdif", data=max_performance_risk, palette="Set2")
plt.xlabel("")
plt.ylabel("Percentage difference of F-score")
plt.xticks(rotation=30)
# sns.set(font_scale=1.5)
plt.tight_layout()
plt.show()
# plt.savefig(f'Plots/boxplot_max_performance_risk.svg', bbox_inches='tight')

