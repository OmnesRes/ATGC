import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score
import pylab as plt
import seaborn as sns
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

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'data' / 'data.pkl', 'rb'))
del D, tcga_maf
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
class_counts = dict(samples['type'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] > 125]
samples = samples.loc[samples['type'].isin(labels_to_use)]

A = samples['type'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot

y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

predictions, y_label, test_idx = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'context_predictions.pkl', 'rb'))
print(np.sum((np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) * y_weights[np.concatenate(test_idx)]))
print(sum(np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) / len(y_label))
print(roc_auc_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), predictions, multi_class='ovr'))


matrix = confusion_matrix(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), (np.argmax(predictions, axis=-1)))
recalls = recall_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), (np.argmax(predictions, axis=-1)), average=None)
precisions = precision_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), (np.argmax(predictions, axis=-1)), average=None)


fig = plt.figure()
fig.subplots_adjust(hspace=.05,
                    wspace=.05,
                    left=.05,
                    right=.97,
                    bottom=.07,
                    top=.95)
gs = fig.add_gridspec(2, 2, width_ratios=[24, 1], height_ratios=[1, 24])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

vmax = np.max(precisions * 100, axis=-1)
myblue = make_colormap({0: '#ffffff', vmax: '#4169E1'})
r_matrix = sns.heatmap(np.around(precisions[np.newaxis, :] * 100, 0).astype(np.int32), annot=True, vmin=0, vmax=vmax, cmap=myblue, ax=ax1, cbar=False, fmt='d', annot_kws={'fontsize': 6})
c_matrix_annot = sns.heatmap(np.around(matrix / matrix.sum(axis=-1, keepdims=True) * 100, 0).astype(np.int32), annot=True, vmin=0, vmax=vmax, cmap=myblue, ax=ax2, cbar=False, fmt='d', annot_kws={'fontsize': 6}, edgecolor='face')
p_matrix = sns.heatmap(np.around(recalls[:, np.newaxis] * 100, 0).astype(np.int32), annot=True, vmin=0, vmax=vmax, cmap=myblue, ax=ax3, cbar=False, fmt='d', annot_kws={'fontsize': 6}, edgecolor='face')

ax2.tick_params(axis='x', length=0, width=0, labelsize=6, rotation=270)
ax2.tick_params(axis='y', length=0, width=0, labelsize=6)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('Precision')
ax1.xaxis.set_label_position('top')
ax2.set_xticks(np.array(range(len(classes))) + .5)
ax2.set_yticks(np.array(range(len(classes))) + .5)
ax2.set_xticklabels(classes)
ax2.set_yticklabels(classes)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel('Recall', rotation=270, labelpad=10)
ax3.yaxis.set_label_position('right')
plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'figures' / 'context_confusion_matrix.png', dpi=600)



predictions, y_label, test_idx = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'gene_predictions.pkl', 'rb'))
print(np.sum((np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) * y_weights[np.concatenate(test_idx)]))
print(sum(np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) / len(y_label))
print(roc_auc_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), predictions, multi_class='ovr'))


matrix = confusion_matrix(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), (np.argmax(predictions, axis=-1)))
recalls = recall_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), (np.argmax(predictions, axis=-1)), average=None)
precisions = precision_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), (np.argmax(predictions, axis=-1)), average=None)


fig = plt.figure()
fig.subplots_adjust(hspace=.05,
                    wspace=.05,
                    left=.05,
                    right=.97,
                    bottom=.07,
                    top=.95)
gs = fig.add_gridspec(2, 2, width_ratios=[24, 1], height_ratios=[1, 24])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

vmax = np.max(precisions * 100, axis=-1)
myblue = make_colormap({0: '#ffffff', vmax: '#4169E1'})
r_matrix = sns.heatmap(np.around(precisions[np.newaxis, :] * 100, 0).astype(np.int32), annot=True, vmin=0, vmax=vmax, cmap=myblue, ax=ax1, cbar=False, fmt='d', annot_kws={'fontsize': 6})
c_matrix_annot = sns.heatmap(np.around(matrix / matrix.sum(axis=-1, keepdims=True) * 100, 0).astype(np.int32), annot=True, vmin=0, vmax=vmax, cmap=myblue, ax=ax2, cbar=False, fmt='d', annot_kws={'fontsize': 6}, edgecolor='face')
p_matrix = sns.heatmap(np.around(recalls[:, np.newaxis] * 100, 0).astype(np.int32), annot=True, vmin=0, vmax=vmax, cmap=myblue, ax=ax3, cbar=False, fmt='d', annot_kws={'fontsize': 6}, edgecolor='face')

ax2.tick_params(axis='x', length=0, width=0, labelsize=6, rotation=270)
ax2.tick_params(axis='y', length=0, width=0, labelsize=6)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('Precision')
ax1.xaxis.set_label_position('top')
ax2.set_xticks(np.array(range(len(classes))) + .5)
ax2.set_yticks(np.array(range(len(classes))) + .5)
ax2.set_xticklabels(classes)
ax2.set_yticklabels(classes)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel('Recall', rotation=270, labelpad=10)
ax3.yaxis.set_label_position('right')
plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'figures' / 'gene_confusion_matrix.png', dpi=600)



