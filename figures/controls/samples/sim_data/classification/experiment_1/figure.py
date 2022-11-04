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

random_forest_metrics, logistic_metrics = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'standard_metrics.pkl', 'rb'))
instance_sum_evaluations, instance_sum_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'instance_model_sum.pkl', 'rb'))
instance_mean_evaluations, instance_mean_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'instance_model_mean.pkl', 'rb'))
sample_sum_attention_evaluations, sample_sum_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_attention_sum.pkl', 'rb'))
sample_mean_attention_evaluations, sample_mean_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_attention_mean.pkl', 'rb'))
sample_dynamic_attention_evaluations, sample_dynamic_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_attention_dynamic.pkl', 'rb'))

accuracies = [i[1] for i in instance_mean_evaluations + instance_sum_evaluations + \
                   sample_mean_attention_evaluations + sample_sum_attention_evaluations + \
                   sample_dynamic_attention_evaluations]

accuracies = random_forest_metrics + logistic_metrics + accuracies

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
colors = [paired[0]] * 3 + [paired[1]] * 3 + [paired[2]] * 3 + [paired[3]] * 3 + [paired[4]] * 3 + [paired[5]] * 3 + [paired[6]] * 3

centers = np.concatenate([np.arange(3) + i * 3.2 for i in range(7)])

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=.978,
bottom=0.023,
left=0.14,
right=.99,
hspace=0.2,
wspace=0.2)
ax.bar(centers, accuracies, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)
ax.set_xlim(min(centers) - .52, max(centers) + .54)
ax.set_ylim(-.01, 1.01)
ax.set_yticks([0, .25, .5, .75, 1], )
ax.tick_params(axis='y', length=5, width=1, labelsize=12, rotation=0)
ax.set_yticklabels([0, 25, 50, 75, 100])
ax.set_xticks([])
ax.set_ylabel('Accuracy', fontsize=24, labelpad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['left'].set_bounds(0, 1)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_visible(False)

plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_1' / 'figure.pdf')


