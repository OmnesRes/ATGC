import pickle
import pylab as plt
import numpy as np
import pathlib
from matplotlib import cm
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]

with open(cwd / 'figures' / 'msi' / 'results' / 'for_mantis_plot.pkl', 'rb') as f:
    predictions, test_idx, sample_df, y_label = pickle.load(f)


y_true = y_label[:, 0][np.concatenate(test_idx)]

##compare atgc, mantis predictions

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.99)
fig.subplots_adjust(left=.08)
fig.subplots_adjust(right=.99)
ax.scatter(np.concatenate([i[:, 0] for i in predictions])[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)][y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 0],
            sample_df['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)].values[y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 0],
           s=sample_df['all_counts'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)][y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 0].values/1000,
           edgecolor='k', linewidths=.1, label='MSI Low', color=paired[1])

scatter = ax.scatter(np.concatenate([i[:, 0] for i in predictions])[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)][y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 1],
            sample_df['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)].values[y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 1],
           s=sample_df['all_counts'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)][y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 1].values/1000,
           edgecolor='k', linewidths=.1, label='MSI High', color=paired[5])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['left'].set_position(['outward', -5])
ax.spines['left'].set_bounds(.2, 1.4)
ax.spines['bottom'].set_linewidth(1)
ax.spines['bottom'].set_bounds(0, 1)
ax.tick_params(axis='x', length=4, width=1, labelsize=8)
ax.tick_params(axis='y', length=4, width=1, labelsize=8)
ax.set_xlabel('ATGC')
ax.set_ylabel('MANTIS')
legend1 = ax.legend(frameon=False, loc=(.15,.88), ncol=2, columnspacing=.2, handletextpad=0)
legend1.legendHandles[0].set_sizes([30])
legend1.legendHandles[1].set_sizes([30])
ax.add_artist(legend1)
handles, labels = scatter.legend_elements(prop='sizes', num=5, alpha=1, color='w', markeredgecolor='k', markeredgewidth=1)
ax.legend(handles, labels, frameon=False, title='TMB', loc=(.03, .59))
plt.savefig(cwd / 'figures' / 'msi' / 'atgc_mantis.pdf')



