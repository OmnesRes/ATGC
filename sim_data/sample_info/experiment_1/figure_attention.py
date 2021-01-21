import pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import pathlib
from matplotlib import cm
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sim_data.pkl', 'rb'))

idx_test, sum_before_attentions, predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_sum_before_eval.pkl', 'rb'))
idx_test, sum_after_attentions, predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_sum_after_eval.pkl', 'rb'))
idx_test, both_before_attentions, predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_both_before_eval.pkl', 'rb'))
idx_test, both_after_attentions, predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_both_after_eval.pkl', 'rb'))
idx_test, dynamic_before_attentions, predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_dynamic_before_eval.pkl', 'rb'))
idx_test, dynamic_after_attentions, predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_dynamic_after_eval.pkl', 'rb'))


idx_test = idx_test[:20]
indexes = [np.where(D['sample_idx'] == idx) for idx in idx_test]

classes = []
for i in indexes:
    classes.append(D['class'][i])


models = np.concatenate(classes).shape[0] * [0] + np.concatenate(classes).shape[0] * [1] + np.concatenate(classes).shape[0] * [2]+\
         np.concatenate(classes).shape[0] * [3] + np.concatenate(classes).shape[0] * [4] + np.concatenate(classes).shape[0] * [5]


classes = np.concatenate([np.concatenate(classes), np.concatenate(classes) + 2, np.concatenate(classes) + 4, np.concatenate(classes) + 6,
                          np.concatenate(classes) + 8, np.concatenate(classes) + 10])

# classes = list(np.concatenate(classes)) * 6

types = list(np.concatenate(np.array([np.array(samples['type'])[D['sample_idx']][i] for i in indexes]))) * 6

attention = np.concatenate([np.concatenate(sum_before_attentions[1][:20]),
                            np.concatenate(sum_after_attentions[1][:20]),
                            np.concatenate(both_before_attentions[1][:20]),
                            np.concatenate(both_after_attentions[1][:20]),
                            np.concatenate(dynamic_before_attentions[1][:20]),
                            np.concatenate(dynamic_after_attentions[1][:20]),
                            ])

instance_df = pd.DataFrame({'attention': attention.flat, 'class': classes, 'model': models, 'type': types})

paired = [cm.get_cmap('Paired')(i) for i in range(12)]
palettes = []
for i in range(12):
    palettes.append({j: paired[i] for j in range(12)})

instance_dfs = []
for i in range(12):
    new_attention = [j if k == i else -j for j, k in zip(instance_df.attention.values, instance_df['class'].values)]
    instance_dfs.append(pd.DataFrame({'attention': new_attention, 'class': classes, 'model': models, 'type': types}))


fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.927,
bottom=0.104,
left=0.043,
right=1.0,
hspace=0.2,
wspace=0.2)
for i in range(12):
    sns.stripplot(x="model", y="attention", hue='type', dodge=True, jitter=.35, data=instance_dfs[i],
                  edgecolor='k', linewidth=1, palette=palettes[i], ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_ylim(0, max(instance_df['attention'].values))
ax.set_xticks([])
ax.tick_params(length=0)
ax.set_ylabel('Attention', fontsize=16)
# ax.legend(frameon=False, title='Sample Type', fontsize=10, title_fontsize=12, loc=(-.05, .95), ncol=3)
ax.get_legend().remove()
# plt.savefig(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'attention.png', dpi=300)

