import pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from matplotlib import cm
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'sim_data.pkl', 'rb'))

idx_test, mean_attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'mean_attentions.pkl', 'rb'))
idx_test, sum_attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'sum_attentions.pkl', 'rb'))
idx_test, both_attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'both_attentions.pkl', 'rb'))
idx_test, dynamic_attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'dynamic_attentions.pkl', 'rb'))


idx_test = idx_test[:20]
indexes = [np.where(D['sample_idx'] == idx) for idx in idx_test]

classes = []
for i in indexes:
    classes.append(D['class'][i])

types = np.concatenate(classes).shape[0] * [0] + np.concatenate(classes).shape[0] * [1] + np.concatenate(classes).shape[0] * [2] + np.concatenate(classes).shape[0] * [3]
classes = np.concatenate([np.concatenate(classes), np.concatenate(classes) + 2, np.concatenate(classes) + 4, np.concatenate(classes) + 6])
attention = np.concatenate([np.concatenate(mean_attentions[2][:20]),
                            np.concatenate(sum_attentions[2][:20]),
                            np.concatenate(both_attentions[2][:20]),
                            np.concatenate(dynamic_attentions[2][:20])])

instance_df = pd.DataFrame({'attention': attention.flat, 'class': classes, 'type': types})

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
palette = {0: paired[0], 1: paired[1], 2: paired[2], 3: paired[3], 4: paired[4], 5: paired[5], 6: paired[6], 7: paired[7]}


fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1.0,
bottom=0.0,
left=0.033,
right=1.0,
hspace=0.2,
wspace=0.2)
sns.stripplot(x="type", y="attention", hue='class', data=instance_df, edgecolor='k', linewidth=1, jitter=.4, palette=palette)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlabel('')
ax.set_ylabel('Attention', fontsize=24, labelpad=-10)
ax.get_legend().remove()
plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'attention.png', dpi=300)

