import pylab as plt
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

D, samples = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_2' / 'sim_data.pkl', 'rb'))

sample_sum_before_evaluations, sample_sum_before_histories, weights = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_2' / 'sample_model_sum_before.pkl', 'rb'))
sample_sum_after_evaluations, sample_sum_after_histories, weights = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_2' / 'sample_model_sum_after.pkl', 'rb'))







losses = np.array([i[-1] for i in sample_sum_before_evaluations + sample_sum_after_evaluations])
losses = losses / max(losses)

epochs = np.array([len(i['val_mse']) - 20 for i in sample_sum_before_histories + sample_sum_after_histories])
epochs = epochs / max(epochs)


colors = ['#1f77b4'] * 6 + ['#ff7f0e'] * 6

spacer = np.ones_like(losses)/25
centers = np.concatenate([np.arange(6) + i * 6.2 for i in range(2)])
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=1.0,
bottom=0.0,
left=0.05,
right=0.945,
hspace=0.2,
wspace=0.2)
ax.bar(centers, losses, edgecolor='k', bottom=spacer, color=colors, align='center', linewidth=.5, width=1)

ax.set_xlim(min(centers) - .503, max(centers) + .503)
ax.set_ylim(-max(epochs) - .003, max(losses + spacer) + .003)
ax.set_yticks([])
ax.set_xticks([])


ax2 = ax.twinx()
ax2.bar(centers, -epochs, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)
ax2.set_ylim(-max(epochs) - .003, max(losses + spacer) + .003)
ax2.set_xlim(min(centers) - .503, max(centers) + .503)
ax2.set_yticks([])
ax2.set_xticks([])

ax.set_ylabel(' ' * 19 + 'Losses', fontsize=24, labelpad=0)
ax2.set_ylabel(' ' * 19 + 'Epochs', rotation=-90, fontsize=24, labelpad=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
plt.savefig(cwd / 'sim_data' / 'sample_info' / 'experiment_2' / 'figure.pdf')


