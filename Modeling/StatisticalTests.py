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
    GR = df[df['solution'] == 'G, R'][col].reset_index(drop=True)
    GS = df[df['solution'] == 'G, S'][col].reset_index(drop=True)
    GN = df[df['solution'] == 'G, N'][col].reset_index(drop=True)
    GT = df[df['solution'] == 'G, T'][col].reset_index(drop=True)
    NR = df[df['solution'] == 'N, R'][col].reset_index(drop=True)
    NS = df[df['solution'] == 'N, S'][col].reset_index(drop=True)
    NT = df[df['solution'] == 'N, T'][col].reset_index(drop=True)
    RS = df[df['solution'] == 'R, S'][col].reset_index(drop=True)
    RT = df[df['solution'] == 'R, T'][col].reset_index(drop=True)
    ST = df[df['solution'] == 'S, T'][col].reset_index(drop=True)
    GNR = df[df['solution'] == 'G, N, R'][col].reset_index(drop=True)
    GNS = df[df['solution'] == 'G, N, S'][col].reset_index(drop=True)
    GNT = df[df['solution'] == 'G, N, T'][col].reset_index(drop=True)
    GRS = df[df['solution'] == 'G, R, S'][col].reset_index(drop=True)
    GRT = df[df['solution'] == 'G, R, T'][col].reset_index(drop=True)
    GST = df[df['solution'] == 'G, S, T'][col].reset_index(drop=True)
    NRS = df[df['solution'] == 'N, R, S'][col].reset_index(drop=True)
    NRT = df[df['solution'] == 'N, R, T'][col].reset_index(drop=True)
    NST = df[df['solution'] == 'N, S, T'][col].reset_index(drop=True)
    RST = df[df['solution'] == 'R, S, T'][col].reset_index(drop=True)
    GNRS = df[df['solution'] == 'G, N, R, S'][col].reset_index(drop=True)
    GNRT = df[df['solution'] == 'G, N, R, T'][col].reset_index(drop=True)
    GNST = df[df['solution'] == 'G, N, S, T'][col].reset_index(drop=True)
    GRST = df[df['solution'] == 'G, R, S, T'][col].reset_index(drop=True)
    NRST = df[df['solution'] == 'N, R, S, T'][col].reset_index(drop=True)
    GNRST = df[df['solution'] == 'G, N, R, S, T'][col].reset_index(drop=True)

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


transfs_name = ['G', 'N', 'R', 'S', 'T', 'G, N', 'G, R', 'G, S',
                'G, T', 'N, R', 'N, S', 'N, T', 'R, S', 'R, T',
                'S, T', 'G, N, R', 'G, N, S', 'G, N, T',
                'G, R, S', 'G, R, T', 'G, S, T', 'N, R, S',
                'N, R, T', 'N, S, T', 'R, S, T', 'G, N, R, S',
                'G, N, R, T', 'G, N, S, T', 'G, R, S, T',
                'N, R, S, T', 'G, N, R, S, T']


# %%
def ploting(res, bbox, file_name):
    res = res[res['Probability'] > 0.0099]
    if res.Result.nunique() != 3:
        palette = ['tab:blue', 'orange']
    else:
        palette = ['tab:blue', 'orange', 'tab:green']
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 3))
    g = sns.histplot(res, x='Solution', hue='Result', weights='Probability', edgecolor='none',
                     multiple='stack', palette=palette, shrink=0.8)
    g.set_ylabel('Probability')
    g.set_xlabel('')
    # fix the legend
    bbox = bbox
    move_legend(g, bbox, "upper center", ax)
    sns.set(font_scale=0.8)
    ax.margins(x=0.02)
    plt.xticks(rotation=30)
    plt.yticks(np.arange(0, 1.25, 0.25))
    plt.axhline(y=0.5, color='lightgrey', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Plots/{file_name}.pdf', bbox_inches='tight')


# %%
def all_func(df, var, bbox, file_name):
    transfs = metric_solution(df, var)

    for i in range(0, len(transfs)):
        transfs[i] = BayesianSignTest(transfs[i], -1, 1)

    solution_res = assign_hyperband(transfs, transfs_name)

    ploting(solution_res, bbox=bbox, file_name=file_name)


# %% Best performance for each solution
# With CV
transf_diff = pd.read_csv('Data/Final_results/AlloutputFiles/Transformed/Testing_18ds/test_results_max_solution.csv',
                          sep='\t')

all_func(transf_diff, 'f1_weighted_perdif', bbox=(0.5, -.25), file_name='prob_solutions_baselineOrig_CV')


# Without CV
transf_diff_no_CV = pd.read_csv('Data/Final_results/AlloutputFiles/Transformed/Testing_18ds/test_results_no_CV.csv',
                                sep='\t')

transf_diff_no_CV = transf_diff_no_CV.groupby(['ds', 'solution'])['f1_weighted_perdif'].max().reset_index()

all_func(transf_diff_no_CV, 'f1_weighted_perdif', bbox=(0.5, -.25), file_name='prob_solutions_baselineOrig_test')

# %% Best solution as baseline for each dataset
# With CV
transf_diff_sol = pd.read_csv('Data/Final_results/AlloutputFiles/Transformed/Testing_18ds'
                              '/test_results_max_solution_diff_bestsol.csv', sep='\t')

all_func(transf_diff_sol, 'f1_weighted_perdif_bestsol', bbox=(0.5, -.25), file_name='prob_solutions_baselineTransf_CV')

# Without CV
transf_diff_sol_no_CV = pd.read_csv('Data/Final_results/AlloutputFiles/Transformed/Testing_18ds'
                                    '/test_results_no_CV_diff_bestsol.csv', sep='\t')

transf_diff_sol_no_CV = transf_diff_sol_no_CV.groupby(['ds', 'solution'])[
    'f1_weighted_perdif_bestsol'].max().reset_index()

all_func(transf_diff_sol_no_CV, 'f1_weighted_perdif_bestsol', bbox=(0.5, -.25), file_name='prob_solutions_baselineTransf_test')


# %% Best protection as baseline for each dataset
# With CV
transf_diff_sol_priv = pd.read_csv('Data/Final_results/AlloutputFiles/Transformed/Testing_18ds'
                                   '/test_results_diff_bestsol_priv.csv', sep='\t')

all_func(transf_diff_sol_priv, 'f1_weighted_perdif_bestsol_priv', bbox=(0.5, -.25), file_name='prob_solutions_baselinePriv_CV')


# Without CV
transf_diff_sol_priv = pd.read_csv('Data/Final_results/AlloutputFiles/Transformed/Testing_18ds'
                                   '/test_results_no_CV_diff_bestsol_priv.csv', sep='\t')

all_func(transf_diff_sol_priv, 'f1_weighted_perdif_bestsol_priv', bbox=(0.5, -.25), file_name='prob_solutions_baselinePriv_test')
