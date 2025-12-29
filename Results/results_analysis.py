import pandas as pd

df = pd.read_csv('Results/test.csv')
df = pd.read_csv('Results/test_fede.csv')


df['ga_nn_gap'] = 100*(1 - df.obj_nn_ga/df.obj_exact)
df['ga_gap'] = 100*(1 - df.obj_ga/df.obj_exact)
df['case'] = df.commodities.astype(int).astype(str) + ' ' + df.paths.astype(int).astype(str)


# df_res = df.groupby('case_num').agg(
#     {'case': 'min', 'ga_nn_gap': ['mean', 'std'], 'ga_gap': ['mean', 'std'], 'time_exact': ['mean', 'std'],
#      'time_h': ['mean', 'std'], 'time_ga': ['mean', 'std']})


df_res = df.groupby('case_num').agg(
    {'case': 'min', 'ga_nn_gap': 'mean', 'ga_gap': 'mean', 'time_exact': 'mean',
     'time_nn_ga': 'mean', 'time_ga': 'mean'})
df_res.columns = ['CASE', 'GAP\% NN+GA', 'GAP\% GA', 'TIME EXACT', 'TIME NN+GA', 'TIME GA']
df_res_ = df_res.reset_index().drop(columns='case_num')
df_tex = df_res_.style.format(decimal=',', thousands='.', precision=2).hide(axis="index")
print(df_tex.to_latex(column_format='c|cc|ccc'))
