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

idx_test, attention, sample_sum_before_predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_2' / 'sample_model_sum_before_attention.pkl', 'rb'))
idx_test, attention, sample_sum_after_predictions = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_2' / 'sample_model_sum_after_attention.pkl', 'rb'))

##get x_true and y_true

x_true = []
indexes = []

for i in range(1, 4):
    x_temp = []
    temp_indexes = []
    for sample_idx in idx_test:
        if sample_idx in np.where(np.array(samples['type']) == i)[0]:
            variants = D['class'][np.where(D['sample_idx'] == sample_idx)]
            x_temp.append(len(np.where(variants != 0)[0]))
            temp_indexes.append(sample_idx)
    x_true.append(x_temp)
    indexes.append(temp_indexes)

predictions = [[np.array(samples['values'])[type_indexes] for type_indexes in indexes],
               [sample_sum_before_predictions[1][np.isin(idx_test, type_indexes)] for type_indexes in indexes],
               [sample_sum_after_predictions[1][np.isin(idx_test, type_indexes)] for type_indexes in indexes]]



z_order = np.arange(0, 100)
colors = ['k', '#1f77b4', '#ff7f0e']
markers = ['o', '^', 's']
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=1.0,
bottom=0.075,
left=0.035,
right=1.0,
hspace=0.2,
wspace=0.2)
for index in range(3):
    if index != 0:
        for index2, (x_type, y_type) in enumerate(zip(x_true, predictions[index])):
            for x, y in zip(x_type, np.exp(y_type) - 1):
                ax.scatter(x, y, color=colors[index], edgecolor='k', linewidth=.5, zorder=np.random.choice(z_order), marker=markers[index2])

    else:
        for index2, (x_type, y_type) in enumerate(zip(x_true, predictions[index])):
            for x, y in zip(x_type, y_type):
                ax.scatter(x, y, color=colors[index], edgecolor='k', linewidth=.5, zorder=np.random.choice(z_order), marker=markers[index2])
#
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(-10, sorted(np.concatenate(x_true))[-10] - 5)
ax.set_ylim(0, sorted(np.concatenate(predictions[0]))[-10])
ax.set_ylabel('Bag Value', fontsize=24, labelpad=-10)
ax.set_xlabel('Key Instance Count', fontsize=24, labelpad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=colors[0], edgecolor='k', label='True Value'),
                   Patch(facecolor=colors[1], edgecolor='k', label='Before'),
                   Patch(facecolor=colors[2], edgecolor='k', label='After')]

ax.legend(handles=legend_elements, loc='upper left', fontsize=14, frameon=False)
plt.savefig(cwd / 'sim_data' / 'sample_info' / 'experiment_2' / 'figure_predictions.png', dpi=300)


