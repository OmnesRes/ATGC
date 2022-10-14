import numpy as np
import pickle
import pylab as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

def make_colormap(colors):
    from matplotlib.colors import LinearSegmentedColormap, ColorConverter
    z = np.sort(list(colors.keys()))
    anchors = (z - min(z)) / (max(z) - min(z))
    CC = ColorConverter()
    R, G, B = [], [], []
    for i in range(len(z)):
        Ci = colors[z[i]]
        RGB = CC.to_rgb(Ci)
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])
    cmap_dict = {}
    cmap_dict['red'] = [(anchors[i], R[i], R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(anchors[i], G[i], G[i]) for i in range(len(G))]
    cmap_dict['blue'] = [(anchors[i], B[i], B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap', cmap_dict)
    return mymap

with open(cwd / 'figures' / 'msi' / 'results' / 'mil_scores.pkl', 'rb') as f:
    mil_recalls, mil_precisions, mil_scores, mil_predictions, samples, y_label, test_idx, train_valids = pickle.load(f)

with open(cwd / 'figures' / 'msi' / 'results' / 'msipred_scores.pkl', 'rb') as f:
    msipred_recalls, msipred_precisions, msipred_scores, msipred_predictions, samples, y_label, test_idx, train_valids = pickle.load(f)

y_true = 1 - y_label[:, 0][np.concatenate(test_idx)]

mil_recalls, mil_precisions, msipred_recalls, msipred_precisions, mantis_recalls, mantis_precisions = [], [], [], [], [], []

for cancer in ['UCEC', 'STAD', 'COAD', 'READ', 'ESCA', 'UCS']:
    mask = samples['type'][np.concatenate(test_idx)] == cancer
    mantis_pred = samples['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)].apply(lambda x: 1 if x > .4 else 0).values[mask[~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)]]
    mil_recalls.append(recall_score(y_true[mask], (np.concatenate(mil_predictions)[:, 0][mask] < 0).astype(np.int32)))
    msipred_recalls.append(recall_score(y_true[mask], np.argmax(np.concatenate(msipred_predictions), axis=-1)[mask]))
    mantis_recalls.append(recall_score(y_true[~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)][mask[~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)]], mantis_pred))
    mil_precisions.append(precision_score(y_true[mask], (np.concatenate(mil_predictions)[:, 0][mask] < 0).astype(np.int32)))
    msipred_precisions.append(precision_score(y_true[mask], np.argmax(np.concatenate(msipred_predictions), axis=-1)[mask]))
    mantis_precisions.append(precision_score(y_true[~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)][mask[~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)]], mantis_pred))


precision_matrix = np.stack([mil_precisions,
                             msipred_precisions,
                             mantis_precisions], axis=0)

recall_matrix = np.stack([mil_recalls,
                             msipred_recalls,
                             mantis_recalls], axis=0)


precision_matrix_normed = precision_matrix / np.sum(precision_matrix, axis=0)
recall_matrix_normed = recall_matrix / np.sum(recall_matrix, axis=0)

fig = plt.figure()
fig.subplots_adjust(left=.117,
                    right=.99,
                    top=.88,
                    bottom=.11,
                    wspace=.023
                    )
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
vmax = np.max(np.max(precision_matrix_normed, axis=-1))
vmin = np.min(np.min(precision_matrix_normed, axis=-1))
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})
sns.heatmap(precision_matrix_normed, annot=np.around(precision_matrix * 100).astype(np.int32), vmin=vmin, vmax=vmax, cmap=myblue, ax=ax1, cbar=False, fmt='d', annot_kws={'fontsize': 10}, square=True)
ax1.tick_params(axis='x', length=0, width=0, labelsize=12, rotation=270)
ax1.tick_params(axis='y', length=0, width=0, labelsize=12, rotation=0)
ax1.set_title('Precisions', fontsize=18)
ax1.set_yticks(np.array(range(3)) + .5)
ax1.set_xticks(np.array(range(6)) + .5)
ax1.set_xticklabels(['UCEC', 'STAD', 'COAD', 'READ', 'ESCA', 'UCS'])
ax1.set_yticklabels(['ATGC', 'MSIpred', 'MANTIS'])

vmax = np.max(np.max(recall_matrix_normed, axis=-1))
vmin = np.min(np.min(recall_matrix_normed, axis=-1))
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})
sns.heatmap(recall_matrix_normed, annot=np.around(recall_matrix * 100).astype(np.int32), vmin=vmin, vmax=vmax, cmap=myblue, ax=ax2, cbar=False, fmt='d', annot_kws={'fontsize': 10}, square=True)
ax2.tick_params(axis='x', length=0, width=0, labelsize=12, rotation=270)
ax2.set_title('Recalls', fontsize=18)
ax2.set_yticks([])
ax2.set_xticks(np.array(range(6)) + .5)
ax2.set_xticklabels(['UCEC', 'STAD', 'COAD', 'READ', 'ESCA', 'UCS'])
plt.savefig(cwd / 'figures' / 'msi' / 'metrics.png', dpi=600)

