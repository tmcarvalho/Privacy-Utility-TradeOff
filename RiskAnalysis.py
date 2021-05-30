# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
combs = pd.read_pickle('Data/Final_results/final_combs.pkl')

# %%
# store the position of the datasets that have the 5 solutions at the same time
lst_all_solutions = []
for idx, comb in enumerate(combs):
    if len(comb) == 31:
        lst_all_solutions.append(idx)  # 18 datasets with all solutions

risk = pd.read_csv('Data/tilemap.csv', sep='\t')
risk['comparisson'] = 'All datasets'

for i in range(0, len(risk)):
    if risk['ds'][i] in lst_all_solutions:
        risk = risk.append(risk.loc[i], ignore_index=True)
        risk.loc[len(risk) - 1, 'comparisson'] = '18 datasets'

risk_mean = risk.groupby(['comb', 'comparisson'])['rl'].mean().reset_index()
risk_mean['rank'] = risk_mean.groupby(['comparisson'])["rl"].rank().astype(int)
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
sns.barplot(data=risk, x="comb", y="diff", hue="comparisson", palette='colorblind')
sns.set_theme(style="whitegrid", font_scale=0.8)
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Percentage difference")
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.18), borderaxespad=0.)
plt.show()
# plt.tight_layout()
# plt.savefig(f'Plots/bodega4.pdf', bbox_inches='tight')

# %% Table with rank
heatmap = pd.pivot_table(risk_mean, values='rank', index=['comb'], columns='comparisson')
row = risk_mean.comb.values.tolist()
row = sorted(set(row), key=custom_key)
heatmap = heatmap.reindex(row)
heatmap.head(n=5)
heatmap = heatmap.T

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(16, 10))
ax = sns.heatmap(heatmap, cmap="YlGnBu", annot=True,
                 cbar_kws=dict(use_gridspec=False, location="bottom", pad=0.1, shrink=0.3,
                               label='Rank of re-identification risk'),
                 square=True)

# fix cbar ticks
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(axis='both', which='both', length=0)

plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=30)
# plt.show()
plt.savefig(f'Plots/bodega50.pdf', bbox_inches='tight')
