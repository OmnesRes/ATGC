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

D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'sample_info' / 'experiment_1' / 'sim_data.pkl', 'rb'))

sample_sum_before_evaluations, sample_sum_before_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_sum_before.pkl', 'rb'))
sample_sum_after_evaluations, sample_sum_after_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_sum_after.pkl', 'rb'))
sample_dynamic_before_evaluations, sample_dynamic_before_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_dynamic_before.pkl', 'rb'))
sample_dynamic_after_evaluations, sample_dynamic_after_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'sample_info' / 'experiment_1' / 'sample_model_dynamic_after.pkl', 'rb'))

losses = [i[-1] for i in sample_sum_before_evaluations + sample_sum_after_evaluations + sample_dynamic_before_evaluations + sample_dynamic_after_evaluations]

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
colors = [paired[0]] * 3 + [paired[1]] * 3 + [paired[2]] * 3 + [paired[3]] * 3

centers = np.concatenate([np.arange(3) + i * 3.2 for i in range(4)])
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=.981,
bottom=0.02,
left=0.142,
right=0.996,
hspace=0.2,
wspace=0.2)
ax.bar(centers, losses, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)

ax.set_xlim(min(centers) - .56, max(centers) + .56)
# ax.set_ylim(-max(epochs) - .003, max(losses + spacer) + .003)
ax.set_yticks([0, .005, .01, .015, .02, .025])
ax.set_xticks([])

ax.set_ylabel('MSE', fontsize=24, labelpad=5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds([0, .025])
ax.spines['bottom'].set_visible(False)
plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'sample_info' / 'experiment_1' / 'figure.pdf')


