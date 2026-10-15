import pandas as pd
import numpy as np
import scipy

#file_name='results_skwp_all.csv'
file_name='test_npp_new.csv'

df = pd.read_csv(file_name, index_col=0)

ga_objs = df[['obj_ga', 'obj_ga_lhc', 'obj_ga_sobol', 'obj_ga_halton', 'obj_ga_kronecker']].to_numpy()
ga_times = df[['time_ga', 'time_ga_lhc', 'time_ga_sobol', 'time_ga_halton', 'time_ga_kronecker']].to_numpy()

best_ga_id = np.arange(len(ga_objs)), np.argmax(ga_objs, axis=1)
best_ga_obj = ga_objs[best_ga_id]
best_ga_time = ga_times[best_ga_id]

df['obj_best_ga'] = best_ga_obj
df['time_best_ga'] = best_ga_time
df['id_best_ga'] = best_ga_id[1]

df['gap_to_gat'] = (df.obj_nn_sample - df.obj_nn_ga) / df.obj_nn_sample
df['gap_to_ga'] = (df.obj_best_ga - df.obj_nn_ga) / df.obj_best_ga
df['gap_to_gurobi'] = (df.obj_exact - df.obj_nn_ga) / df.obj_exact

for _gap, _c_name in [(df.gap_to_gat, 'p_gap_to_gat'), (df.gap_to_ga, 'p_gap_to_ga'), (df.gap_to_gurobi, 'p_gap_to_gurobi')]:                                                             
    _gap = _gap.to_numpy().reshape((-1, 30))
    #p_val = scipy.stats.mannwhitneyu(_gap, np.zeros_like(_gap), alternative='less', axis=1)[1]
    p_val = scipy.stats.wilcoxon(_gap, alternative='less', axis=1)[1]
    p_val = np.repeat(p_val, 30)
    df[_c_name] = p_val 

grouped = df.groupby(['commodities', 'paths']).mean()
#grouped = df.groupby(['M', 'K']).mean()
       

for r_i in zip(list(grouped.index), grouped.gap_to_gat, grouped.p_gap_to_gat, grouped.gap_to_ga, grouped.p_gap_to_ga, grouped.gap_to_gurobi, grouped.p_gap_to_gurobi, grouped.mip_GAP, grouped.time_nn, grouped.time_nn_ga, grouped.time_best_ga, grouped.time_exact):
    (c, p), g_gat, p_gat, g_ga, p_ga, g_gb, p_gb, g_mip, t_gat, t_nn_ga, t_ga, t_gb = r_i
    print(f'{int(c)} & {int(p)} & {g_gat*100:.2f} ({p_gat:.3f}) & {g_ga*100:.2f} ({p_ga:.3f}) & {g_gb*100:.2f} ({p_gb:.3f}) & {g_mip*100:.2f} & {t_gat:.2f} & {t_nn_ga:.2f} & {t_ga:.2f} & {t_gb:.2f}\\\\')

