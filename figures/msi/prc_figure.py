import pickle
import pylab as plt
import numpy as np
import pathlib
from matplotlib import cm
from sklearn.metrics import average_precision_score, precision_recall_curve
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

with open(cwd / 'figures' / 'msi' / 'results' / 'mil_scores.pkl', 'rb') as f:
    mil_recalls, mil_precisions, mil_scores, predictions, samples, y_label, test_idx, train_valids = pickle.load(f)

with open(cwd / 'figures' / 'msi' / 'results' / 'msipred_scores.pkl', 'rb') as f:
    msipred_recalls, msipred_precisions, msipred_scores, predictions, samples, y_label, test_idx, train_valids = pickle.load(f)

y_true = 1 - y_label[:, 0][np.concatenate(test_idx)]
paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.1)
fig.subplots_adjust(top=1)
fig.subplots_adjust(left=.087)
fig.subplots_adjust(right=1)
average_precision = np.sum(np.array(mil_scores) * np.array([i.shape[0] for i in test_idx]) / len(y_true))
ax.plot(np.mean(mil_recalls, axis=0), np.mean(mil_precisions, axis=0), linewidth=1, label=f"{'ATGC:':>13}{f'{average_precision:.3f}':>9}", color=paired[1])
average_precision = np.sum(np.array(msipred_scores) * np.array([i.shape[0] for i in test_idx]) / len(y_true))
ax.plot(np.mean(msipred_recalls, axis=0), np.mean(msipred_precisions, axis=0), linewidth=1, label=f"{'MSIpred:':>12}{f'{average_precision:.3f}':>9}", color=paired[3])
precision, recall, _ = precision_recall_curve(y_true[~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)], samples['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)].values)
average_precision = average_precision_score(y_true[~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)], samples['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)].values)
ax.plot(recall, precision, linewidth=1, color=paired[5], label=f"{'MANTIS:':>12} {f'{average_precision:.3f}':>8}")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['left'].set_bounds(.2, 1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['bottom'].set_bounds(0, 1)
ax.tick_params(axis='x', length=5, width=1, labelsize=8)
ax.tick_params(axis='y', length=5, width=1, labelsize=8)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.legend(fontsize=8, borderpad=0, title='Average Precision Score', title_fontsize=10, frameon=False, loc=(.04, .78))
plt.savefig(cwd / 'figures' / 'msi' / 'prc.pdf')





