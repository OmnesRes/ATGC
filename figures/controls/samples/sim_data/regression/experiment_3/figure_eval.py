import pylab as plt
import numpy as np
import pickle
import pathlib
from matplotlib import cm
path = pathlib.Path.cwd()

if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'sim_data.pkl', 'rb'))

sample_sum_attention_evaluations, sample_sum_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'sample_model_attention_sum.pkl', 'rb'))
sample_mean_attention_evaluations, sample_mean_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'sample_model_attention_mean.pkl', 'rb'))
sample_dynamic_attention_evaluations, sample_dynamic_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'sample_model_attention_dynamic.pkl', 'rb'))


losses = np.array([i[-1] for i in sample_mean_attention_evaluations + sample_sum_attention_evaluations + \
                   sample_dynamic_attention_evaluations])

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
colors = [paired[0]] * 3 + [paired[1]] * 3 + [paired[2]] * 3

centers = np.concatenate([np.arange(3) + i * 3.2 for i in range(3)])

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=.99,
bottom=0.032,
left=0.12,
right=0.99,
hspace=0.2,
wspace=0.2)
ax.bar(centers, losses, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)

ax.set_xlim(min(centers) - .6, max(centers) + .53)
ax.set_ylim(0, .08)
ax.set_xticks([])
ax.set_ylabel('MSE', fontsize=24, labelpad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'figure.pdf')


