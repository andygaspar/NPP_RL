import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_csv('Results/skwp_results.csv', na_values=['inf', '-inf', 'Infinity'])
# df = pd.read_csv('Results/test_npp_20_90_120526.csv', na_values=['inf', '-inf', 'Infinity'])

# df = df[df.commodities > 1]
df['instance'] = df.commodities.astype(int).astype(str) + ' ' + df.paths.astype(int).astype(str)

inst_dict = dict(zip(df['instance'].unique(), range(df['instance'].unique().shape[0])))

df['nn_vs_ga'] = (df.obj_nn_ga - df.obj_ga) /df.obj_ga

mpl.rcParams['figure.figsize'] = (20, 20)
mpl.rcParams['font.size'] = 30

plt.scatter([inst_dict[i] for i in df.instance], df.nn_vs_ga, s=100)
plt.plot(range(len(inst_dict)), np.zeros(len(inst_dict)))
plt.xticks(range(len(inst_dict)), list(inst_dict.keys()), rotation=45)
plt.ylabel('GAT + GA vs GS improvement')
plt.tight_layout()
plt.show()

print((df.nn_vs_ga[df.nn_vs_ga > 1].mean() - 1)*100, df.nn_vs_ga[df.nn_vs_ga > 1].shape[0])
print((df.nn_vs_ga[df.nn_vs_ga < 1].mean() -1)*100, df.nn_vs_ga[df.nn_vs_ga < 1].shape[0])
