import pylab as plt
import seaborn as sns
import pickle
import pathlib
from matplotlib import cm
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]

D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sim_data.pkl', 'rb'))

sample_sum_attention_indexes, sample_sum_attention_ranks, sample_sum_attention_sample_dfs, attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_sum_eval.pkl', 'rb'))
sample_dynamic_attention_indexes, sample_dynamic_attention_ranks, sample_dynamic_attention_sample_dfs, attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_dynamic_eval.pkl', 'rb'))

for i, j in zip(sample_sum_attention_sample_dfs, sample_dynamic_attention_sample_dfs):
    i['Model'] = 'Sum'
    j['Model'] = 'Dynamic'

sample_df = sample_sum_attention_sample_dfs[4].append(sample_dynamic_attention_sample_dfs[4])

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
palette = {'Sum': paired[1], 'Dynamic': paired[3]}

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1.0,
bottom=0.062,
left=0.05,
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

plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'figure_predictions.pdf')

