from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sim_data.pkl', 'rb'))

instance_indexes, instance_ranks, instance_sample_df = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'instance_model_sum_eval.pkl', 'rb'))
sample_indexes, sample_ranks, sample_sample_df = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_sum_eval.pkl', 'rb'))

instance_sample_df['Model'] = 'Instance'
sample_sample_df['Model'] = 'Sample'

sample_df = instance_sample_df.append(sample_sample_df)


fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1.0,
bottom=0.055,
left=0.045,
right=1.0,
hspace=0.2,
wspace=0.2)
ax = sns.stripplot(x="class", y="predictions", hue='Model', dodge=True, jitter=.35, data=sample_df, edgecolor='k', linewidth=1, ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.set_ylabel('Predicted Risk', fontsize=16)
ax.set_xlabel('Bag Label', fontsize=16)
ax.legend(frameon=False, title='Model', fontsize=12, title_fontsize=14, loc=(.01, .8))


plt.savefig(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'figure.pdf')

