import pylab as plt
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

D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'sim_data.pkl', 'rb'))

idx_test, instance_sum_predictions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'instance_model_sum_predictions.pkl', 'rb'))
idx_test, instance_mean_predictions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'instance_model_mean_predictions.pkl', 'rb'))
idx_test, sample_sum_predictions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'sample_model_sum_predictions.pkl', 'rb'))
idx_test, sample_mean_predictions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'sample_model_mean_predictions.pkl', 'rb'))

##get x_true and y_true

x_true = []
for sample_idx in idx_test:
    variants = D['class'][np.where(D['sample_idx'] == sample_idx)]
    x_true.append(np.ceil(len(np.where(variants != 0)[0]) * 100 / len(np.where(D['sample_idx'] == sample_idx)[0])))

y_true = np.array(samples['values'])[idx_test]


predictions = [y_true, instance_mean_predictions, instance_sum_predictions, sample_mean_predictions, sample_sum_predictions]


z_order = np.arange(0, len(x_true))
paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
colors = ['k', paired[1], paired[3], paired[5], paired[7]]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=1.0,
bottom=0.075,
left=0.035,
right=1.0,
hspace=0.2,
wspace=0.2)
for index, i in enumerate(['True Value', 'Instance Mean', 'Instance Sum', 'Sample Mean', 'Sample Sum']):
    if index != 0:
        for index2, (x, y) in enumerate(zip(x_true, np.exp(predictions[index][0]) - 1)):
            if index2 == len(x_true) - 1:
                ax.scatter(x, y, color=colors[index], edgecolor='k', linewidths=.5, zorder=np.random.choice(z_order), label=i)
            else:
                ax.scatter(x, y, color=colors[index], edgecolor='k', linewidths=.5, zorder=np.random.choice(z_order))

    else:
        for index2, (x, y) in enumerate(zip(x_true, predictions[index])):
            if index2 == len(x_true) - 1:
                ax.scatter(x, y, color=colors[index], linewidths=.5, zorder=np.random.choice(z_order), label=i)
            else:
                ax.scatter(x, y, color=colors[index], linewidths=.5, zorder=np.random.choice(z_order))

ax.set_yticks([])
ax.set_xticks([])

ax.set_ylabel('Bag Value', fontsize=24, labelpad=-10)
ax.set_xlabel('Key Instance Fraction', fontsize=24, labelpad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.legend(frameon=False, loc='upper left', fontsize=14)

plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_3' / 'figure_predictions.png', dpi=300)


