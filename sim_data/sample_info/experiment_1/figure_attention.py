import pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sim_data.pkl', 'rb'))

idx_test, attentions, predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_sum_before_attention.pkl', 'rb'))

idx_test = idx_test[:20]
indexes = [np.where(D['sample_idx'] == idx) for idx in idx_test]

classes = []
for i in indexes:
    classes.append(D['class'][i])

classes = np.concatenate(classes)
types = np.concatenate(np.array([np.array(samples['type'])[D['sample_idx']][i] for i in indexes]))
attention = (np.concatenate(attentions[3][:20]))

instance_df = pd.DataFrame({'attention': attention.flat, 'class': classes, 'type': types})

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.927,
bottom=0.104,
left=0.043,
right=1.0,
hspace=0.2,
wspace=0.2)
sns.stripplot(x="class", y="attention", hue='type', dodge=True, jitter=.35, data=instance_df, edgecolor='k', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.tick_params(length=0)
ax.set_xticklabels(['False', 'True'], fontsize=14)
ax.set_ylabel('Attention', fontsize=16)
ax.set_xlabel('Key Instance', fontsize=16)
ax.legend(frameon=False, title='Sample Type', fontsize=10, title_fontsize=12, loc=(-.05, .95), ncol=3)
ax.set_title('Before', fontsize=24)

plt.savefig(cwd / 'sim_data' / 'sample_info' / 'experiment_1' / 'before_attention.png', dpi=300)

