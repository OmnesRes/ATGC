import pylab as plt
import numpy as np
import pickle
import pathlib
path = pathlib.Path.cwd()
from matplotlib import cm

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'sample_info' / 'experiment_2' / 'sim_data.pkl', 'rb'))


##get x_true and y_true

x_true = []
indexes = []

for i in range(1, 4):
    x_temp = []
    temp_indexes = []
    for sample_idx in range(len(samples['type'])):
        if sample_idx in np.where(np.array(samples['type']) == i)[0]:
            variants = D['class'][np.where(D['sample_idx'] == sample_idx)]
            x_temp.append(len(np.where(variants != 0)[0]))
            temp_indexes.append(sample_idx)
    x_true.append(x_temp)
    indexes.append(temp_indexes)

predictions = [np.array(samples['values'])[type_indexes] for type_indexes in indexes]


paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=1.0,
bottom=0.05,
left=0.035,
right=1.0,
hspace=0.2,
wspace=0.2)
for index in range(3):
    ax.scatter(x_true[index], predictions[index], color=paired[index * 2 + 1], edgecolor='k', linewidth=.5, label='Sample Type ' + str(index + 1))

ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(-1, sorted(np.concatenate(x_true))[-1] - 2)
ax.set_ylim(-1, sorted(np.concatenate(predictions))[-1] - 10)
ax.set_ylabel('Bag Value', fontsize=16, labelpad=0)
ax.set_xlabel('Key Instance Count', fontsize=16, labelpad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.legend(loc=(.01, .8), fontsize=12, frameon=False)
plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'sample_info' / 'experiment_2' / 'figure_predictions.png', dpi=300)


