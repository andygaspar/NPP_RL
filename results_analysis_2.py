import pandas as pd
import numpy as np
import scipy

# (file, case columns, header labels, caption, label, appendix caption, appendix label)
FILES = [
    ('Results/test_npp_new_net.csv', ['commodities', 'paths'], ['C', 'P'],
     'Performance comparison between Gurobi, GAT+GA, and GA.', 'tab:results',
     'Comparison of different GA initialisations on PP', 'tab:ga_npp'),
    ('Results/results_skwp_k1_net.csv', ['M', 'K'], ['M', 'K'],
     'Results on SKWP with $K = 1$, comparison between GAT + GA and plain GA.', 'tab:skwp_k1',
     'Comparison of different GA initialisations on SKWP with $K = 1$', 'tab:ga_skwp_k1'),
    ('Results/results_skwp_mf_net.csv', ['M', 'K'], ['M', 'K'],
     'Results on SKWP with $K > 1$, comparison between GAT + GA and plain GA.', 'tab:skwp',
     'Comparison of different GA initialisations on SKWP with $K > 1$', 'tab:ga_skwp'),
]
CAPTION_TAIL = ' Bracketed values are p-values of the one-sided Wilcoxon signed-rank test on the paired gaps.'


def p_value(gap):
    # one-sided Wilcoxon signed-rank test: H1 gap < 0 (GAT+GA better)
    gap = gap.dropna()
    if len(gap) == 0:
        return np.nan
    try:
        return scipy.stats.wilcoxon(gap, alternative='less')[1]
    except ValueError:
        # all differences are zero
        return np.nan


def fmt_gap(g, p):
    if np.isnan(g):
        return '- (-)'
    return f'{g * 100:.2f} ({p:.2f})' if not np.isnan(p) else f'{g * 100:.2f} (-)'


def fmt(v):
    return '-' if np.isnan(v) else f'{v:.2f}'


appendix = []
for file_name, case_cols, labels, caption, label, app_caption, app_label in FILES:
    df = pd.read_csv(file_name)
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])

    ga_cols = ['obj_ga', 'obj_ga_lhc', 'obj_ga_sobol', 'obj_ga_halton', 'obj_ga_kronecker']
    ga_objs = df[ga_cols].to_numpy()
    ga_times = df[['time_ga', 'time_ga_lhc', 'time_ga_sobol', 'time_ga_halton', 'time_ga_kronecker']].to_numpy()

    best_ga_id = np.arange(len(ga_objs)), np.argmax(ga_objs, axis=1)
    df['obj_best_ga'] = ga_objs[best_ga_id]
    df['time_best_ga'] = ga_times[best_ga_id]
    df['id_best_ga'] = best_ga_id[1]

    # missing exact solutions are stored as -1
    has_exact = df.obj_exact > 0
    df.loc[~has_exact, ['obj_exact', 'mip_GAP', 'time_exact']] = np.nan

    df['gap_to_gat'] = (df.obj_nn_sample - df.obj_nn_ga) / df.obj_nn_sample
    df['gap_to_ga'] = (df.obj_best_ga - df.obj_nn_ga) / df.obj_best_ga
    df['gap_to_gurobi'] = (df.obj_exact - df.obj_nn_ga) / df.obj_exact

    rows, app_rows = [], []
    for _, g in df.groupby('case_num', sort=True):
        c, p = g[case_cols].iloc[0]
        # appendix: mean gap to each GA initialisation (fraction of runs in which it was the best of the five)
        app_rows.append(' & '.join(
            [str(int(c)), str(int(p))] +
            [f'{((g[col] - g.obj_nn_ga) / g[col]).mean() * 100:.2f} ({(g.id_best_ga == j).mean():.2f})'
             for j, col in enumerate(ga_cols)]) + '\\\\')
        rows.append(' & '.join([
            str(int(c)), str(int(p)),
            fmt_gap(g.gap_to_gat.mean(), p_value(g.gap_to_gat)),
            fmt_gap(g.gap_to_ga.mean(), p_value(g.gap_to_ga)),
            fmt_gap(g.gap_to_gurobi.mean(), p_value(g.gap_to_gurobi)),
            fmt(g.mip_GAP.mean() * 100),
            fmt(g.time_nn.mean()), fmt(g.time_nn_ga.mean()), fmt(g.time_best_ga.mean()), fmt(g.time_exact.mean()),
        ]) + '\\\\')

    print('\n\n%', file_name)
    print('\\setlength{\\tabcolsep}{4pt}')
    print('\\begin{table}')
    print('\\begin{center}')
    print('\\caption{' + caption + CAPTION_TAIL + '}')
    print('\\label{' + label + '}')
    print('\\begin{tabular}{c c|ccc||c||cccc}')
    print('\\hline')
    print(' &  & \\multicolumn{3}{c||}{GAP OF GAT + GA VS (\\%)} & \\multicolumn{1}{c||}{MILP GAP (\\%)} & \\multicolumn{4}{c}{TIME (s)} \\\\')
    print(f'{labels[0]} & {labels[1]} & GAT & GA-best & Gurobi & Gurobi & GAT & GAT+GA & GA & Gurobi \\\\')
    print('\\hline')
    print('\n'.join(rows))
    print('\\hline')
    print('\\end{tabular}')
    print('')
    print('\\end{center}')
    print('\\end{table}')
    print('\\setlength{\\tabcolsep}{1.4pt}')

    appendix.append('\n'.join([
        '\n\n% ' + file_name + ' (appendix)',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{table}',
        '\\begin{center}',
        '\\caption{' + app_caption + '}',
        '\\label{' + app_label + '}',
        '\\begin{tabular}{c c|ccccc}',
        '\\hline',
        ' &  & \\multicolumn{5}{c}{GAP GAT + GA VS(\\%)}\\\\',
        f'{labels[0]} & {labels[1]} & Random & LHC & Sobol & Halton & Kronecker\\\\',
        '\\hline',
        '\n'.join(app_rows),
        '\\hline',
        '\\end{tabular}',
        '\\end{center}',
        '\\end{table}',
        '\\setlength{\\tabcolsep}{1.4pt}']))

print('\n'.join(appendix))
