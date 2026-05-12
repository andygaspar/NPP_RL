import pandas as pd

df = pd.read_csv('NPP/Results/test_fede.csv')

df['ga_ga_nn_gap'] = 100*(1 - df.obj_nn_ga/df.obj_exact)
df['ga_ga_gap'] = 100*(1 - df.obj_ga/df.obj_exact)
df['mip_GAP'] = df['mip_GAP'] * 100

df['case'] = df.commodities.astype(int).astype(str) + ' ' + df.paths.astype(int).astype(str)

df_res = df.groupby('case_num').agg(
    {'case': 'min', 'ga_ga_nn_gap': 'mean', 'ga_ga_gap': 'mean', 'mip_GAP': 'mean', 'time_exact': 'mean', 'time_nn_ga': 'mean', 'time_ga': 'mean'})

df_res.columns = ['C P', 'GAP\% NN+GA', 'GAP\% GA', 'MIP GAP\% ', 'TIME EXACT', 'TIME NN+GA', 'TIME GA']
df_res_ = df_res.style.format(precision=2)

latex_code = df_res_.hide(axis='index').to_latex(
    caption="Population size analysis",
    position='h!',  # posizionamento in LaTeX
    column_format='c|cc||c|ccc'
)
print(latex_code)
