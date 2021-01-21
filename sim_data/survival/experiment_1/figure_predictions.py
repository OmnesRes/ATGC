from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import pathlib
from matplotlib import cm
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

instance_sum_indexes, instance_sum_ranks, instance_sum_sample_dfs = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'instance_model_sum_eval.pkl', 'rb'))
sample_sum_indexes, sample_sum_ranks, sample_sum_sample_dfs = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_sum_eval.pkl', 'rb'))
sample_sum_attention_indexes, sample_sum_attention_ranks, sample_sum_attention_sample_dfs, attentions = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_sum_eval.pkl', 'rb'))
sample_both_attention_indexes, sample_both_attention_ranks, sample_both_attention_sample_dfs, attentions = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_both_eval.pkl', 'rb'))
sample_dynamic_attention_indexes, sample_dynamic_attention_ranks, sample_dynamic_attention_sample_dfs, attentions = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_dynamic_eval.pkl', 'rb'))


for i, j, k, l, m in zip(instance_sum_sample_dfs, sample_sum_sample_dfs, sample_sum_attention_sample_dfs, sample_both_attention_sample_dfs, sample_dynamic_attention_sample_dfs):
    i['Model'] = 'Instance Sum'
    j['Model'] = 'Sample Sum'
    k['Model'] = 'Sample Attention Sum'
    l['Model'] = 'Sample Attention Both'
    m['Model'] = 'Sample Attention Dynamic'

sample_df = instance_sum_sample_dfs[4].append(sample_sum_sample_dfs[4]).append(sample_sum_attention_sample_dfs[4]).append(sample_both_attention_sample_dfs[3]).append(sample_dynamic_attention_sample_dfs[4])

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
palette = {'Instance Sum': paired[1], 'Sample Sum': paired[3], 'Sample Attention Sum': paired[5], 'Sample Attention Both': paired[7], 'Sample Attention Dynamic': paired[9]}



fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1.0,
bottom=0.055,
left=0.045,
right=1.0,
hspace=0.2,
wspace=0.2)
sns.stripplot(x="class", y="predictions", hue='Model', dodge=True, jitter=.35, data=sample_df, edgecolor='k', linewidth=1, ax=ax, palette=palette)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('Predicted Log Risk', fontsize=16)
ax.set_xlabel('Bag Label', fontsize=16)
ax.legend(frameon=False, title='Model', fontsize=11, title_fontsize=13, loc=(.01, .7))

plt.savefig(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'figure_predictions.pdf')

