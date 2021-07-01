import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def move_legend(g, bbox, new_loc, ax, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    g.legend(handles, labels, ncol=3, loc=new_loc, bbox_to_anchor=bbox, frameon=False, **kws)


# %%
def metric_solution(df, col):
    G = df[df['solution'] == 'G'][col].reset_index(drop=True)
    N = df[df['solution'] == 'N'][col].reset_index(drop=True)
    R = df[df['solution'] == 'R'][col].reset_index(drop=True)
    S = df[df['solution'] == 'S'][col].reset_index(drop=True)
    T = df[df['solution'] == 'T'][col].reset_index(drop=True)
    GR = df[df['solution'] == 'GR'][col].reset_index(drop=True)
    GS = df[df['solution'] == 'GS'][col].reset_index(drop=True)
    GN = df[df['solution'] == 'GN'][col].reset_index(drop=True)
    GT = df[df['solution'] == 'GT'][col].reset_index(drop=True)
    NR = df[df['solution'] == 'NR'][col].reset_index(drop=True)
    NS = df[df['solution'] == 'NS'][col].reset_index(drop=True)
    NT = df[df['solution'] == 'NT'][col].reset_index(drop=True)
    RS = df[df['solution'] == 'RS'][col].reset_index(drop=True)
    RT = df[df['solution'] == 'RT'][col].reset_index(drop=True)
    ST = df[df['solution'] == 'ST'][col].reset_index(drop=True)
    GNR = df[df['solution'] == 'GNR'][col].reset_index(drop=True)
    GNS = df[df['solution'] == 'GNS'][col].reset_index(drop=True)
    GNT = df[df['solution'] == 'GNT'][col].reset_index(drop=True)
    GRS = df[df['solution'] == 'GRS'][col].reset_index(drop=True)
    GRT = df[df['solution'] == 'GRT'][col].reset_index(drop=True)
    GST = df[df['solution'] == 'GST'][col].reset_index(drop=True)
    NRS = df[df['solution'] == 'NRS'][col].reset_index(drop=True)
    NRT = df[df['solution'] == 'NRT'][col].reset_index(drop=True)
    NST = df[df['solution'] == 'NST'][col].reset_index(drop=True)
    RST = df[df['solution'] == 'RST'][col].reset_index(drop=True)
    GNRS = df[df['solution'] == 'GNRS'][col].reset_index(drop=True)
    GNRT = df[df['solution'] == 'GNRT'][col].reset_index(drop=True)
    GNST = df[df['solution'] == 'GNST'][col].reset_index(drop=True)
    GRST = df[df['solution'] == 'GRST'][col].reset_index(drop=True)
    NRST = df[df['solution'] == 'NRST'][col].reset_index(drop=True)
    GNRST = df[df['solution'] == 'GNRST'][col].reset_index(drop=True)

    transfs = [G, N, R, S, T, GR, GS, GN, GT, NR, NS, NT, RS, RT,
               ST, GNR, GNS, GNT, GRS, GRT, GST, NRS, NRT, NST, RST, GNRS,
               GNRT, GNST, GRST, NRST, GNRST]

    return transfs


def assign_hyperband(df, transfs_name):
    solution_res = pd.DataFrame(columns=['Solution', 'Result', 'Probability'])

    c = 0
    for j in range(0, 3):
        for i in range(0, len(df)):
            c += 1
            if j == 0:
                solution_res.loc[c] = [transfs_name[i], 'Lose', df[i][j]]
            elif j == 1:
                solution_res.loc[c] = [transfs_name[i], 'Draw', df[i][j]]
            else:
                solution_res.loc[c] = [transfs_name[i], 'Win', df[i][j]]
    return solution_res


transfs_name = ['G', 'N', 'R', 'S', 'T', 'GN', 'GR', 'GS',
                'GT', 'NR', 'NS', 'NT', 'RS', 'RT',
                'ST', 'GNR', 'GNS', 'GNT', 'GRS', 'GRT', 'GST', 'NRS',
                'NRT', 'NST', 'RST', 'GNRS','GNRT', 'GNST', 'GRST',
                'NRST', 'GNRST']


# %%
def ploting(res1, res2, bbox, file_name):
    def proc(df):
        df = df[df['Probability'] > 0.0099]
        if df.Result.nunique() != 3:
            palette = ['tab:blue', 'orange']
        else:
            palette = ['tab:blue', 'orange', 'tab:green']
        return df, palette

    res1, pl1 = proc(res1)
    res2, pl2 = proc(res2)
    sns.set_style("darkgrid")
    # fig, ax = plt.subplots(figsize=(8, 3))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    sns.histplot(res1, x='Solution', hue='Result', weights='Probability', edgecolor='none',
                     multiple='stack', palette=pl1, shrink=0.8, ax=ax1, legend=False)
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('')
    ax1.margins(x=0.02)
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)

    g = sns.histplot(res2, x='Solution', hue='Result', weights='Probability', edgecolor='none',
                 multiple='stack', palette=pl1, shrink=0.8, ax=ax2)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('')
    # fix the legend
    bbox = bbox
    move_legend(g, bbox, "upper center", ax2)
    sns.set(font_scale=0.8)
    ax2.margins(x=0.02)
    # plt.xticks(rotation=30)
    # ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
    for ax in fig.axes:
        ax.tick_params(axis='x', labelrotation=30)

    plt.yticks(np.arange(0, 1.25, 0.25))
    plt.axhline(y=0.5, color='lightgrey', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Plots/{file_name}.pdf', bbox_inches='tight')


# %%
def all_func(df1, df2, var, bbox, file_name):
    transfs = metric_solution(df1, var)
    transfs2 = metric_solution(df2, var)

    for i in range(0, len(transfs)):
        transfs[i] = BayesianSignTest(transfs[i], -1, 1)
        transfs2[i] = BayesianSignTest(transfs2[i], -1, 1)

    solution_res = assign_hyperband(transfs, transfs_name)
    solution_res2 = assign_hyperband(transfs2, transfs_name)

    ploting(solution_res, solution_res2, bbox=bbox, file_name=file_name)


# %% Best performance for each solution
# With CV
transf_diff = pd.read_csv('Data/Final_results/AlloutputFiles/Transformed/Testing_18ds/test_results_max_solution.csv',
                          sep='\t')

transf_diff_no_CV = pd.read_csv('Data/Final_results/AlloutputFiles/Transformed/Testing_18ds/test_results_no_CV.csv',
                                sep='\t')

# all_func(transf_diff, 'f1_weighted_perdif', bbox=(0.5, -.25), file_name='prob_solutions_baselineOrig_CV')
# Without CV
transf_diff_no_CV_v1 = transf_diff_no_CV.groupby(['ds', 'solution'])['f1_weighted_perdif'].max().reset_index()

all_func(transf_diff, transf_diff_no_CV_v1, 'f1_weighted_perdif', bbox=(0.5, -.35), file_name='prob_solutions_baselineOrig')

# %% Best solution as baseline for each dataset
transf_diff_no_CV_v2 = transf_diff_no_CV.groupby(['ds', 'solution'])[
    'f1_weighted_perdif_bestsol'].max().reset_index()

all_func(transf_diff, transf_diff_no_CV_v2, 'f1_weighted_perdif_bestsol', bbox=(0.5, -.25), file_name='prob_solutions_baselineTransf')


# %% Best protection as baseline for each dataset
transf_diff_no_CV_v3 = transf_diff_no_CV.groupby(['ds', 'solution'])[
    'f1_weighted_perdif_bestsol_priv'].max().reset_index()

all_func(transf_diff, transf_diff_no_CV_v3, 'f1_weighted_perdif_bestsol_priv', bbox=(0.5, -.25), file_name='prob_solutions_baselinePriv')
