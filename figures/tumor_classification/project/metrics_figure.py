import numpy as np
import pickle
import pylab as plt
import seaborn as sns
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
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

context_mil_precisions, context_mil_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'context_metrics.pkl', 'rb'))
gene_mil_precisions, gene_mil_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'gene_metrics.pkl', 'rb'))
context_net_precisions, context_net_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'context_metrics.pkl', 'rb'))
gene_net_precisions, gene_net_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'gene_metrics.pkl', 'rb'))
context_forest_precisions, context_forest_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'context_forest_metrics.pkl', 'rb'))
gene_forest_precisions, gene_forest_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'gene_forest_metrics.pkl', 'rb'))
context_logistic_precisions, context_logistic_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'context_logistic_metrics.pkl', 'rb'))
gene_logistic_precisions, gene_logistic_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'gene_logistic_metrics.pkl', 'rb'))

context_precision_matrix = np.stack([context_logistic_precisions,
                                     context_forest_precisions,
                                     context_net_precisions,
                                     context_mil_precisions], axis=0)

context_recall_matrix = np.stack([context_logistic_recalls,
                                  context_forest_recalls,
                                  context_net_recalls,
                                  context_mil_recalls], axis=0)

gene_precision_matrix = np.stack([gene_logistic_precisions,
                                  gene_forest_precisions,
                                  gene_net_precisions,
                                  gene_mil_precisions], axis=0)

gene_recall_matrix = np.stack([gene_logistic_recalls,
                               gene_forest_recalls,
                               gene_net_recalls,
                               gene_mil_recalls], axis=0)



context_precision_matrix_normed = context_precision_matrix / np.sum(context_precision_matrix, axis=0)
context_recall_matrix_normed = context_recall_matrix / np.sum(context_recall_matrix, axis=0)

fig = plt.figure()
fig.subplots_adjust(left=.07,
                    right=.99,
                    )
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
vmax = np.max(np.max(context_precision_matrix_normed, axis=-1))
vmin = np.min(np.min(context_precision_matrix_normed, axis=-1))
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})
sns.heatmap(context_precision_matrix_normed, annot=np.around(context_precision_matrix * 100).astype(np.int32), vmin=vmin, vmax=vmax, cmap=myblue, ax=ax1, cbar=False, fmt='d', annot_kws={'fontsize': 10}, square=True)
ax1.tick_params(axis='x', length=0, width=0, labelsize=8, rotation=270)
ax1.tick_params(axis='y', length=0, width=0, labelsize=8, rotation=0)
ax1.set_title('Context Precisions')
ax1.set_yticks(np.array(range(4)) + .5)
ax1.set_xticks(np.array(range(24)) + .5)
ax1.set_xticklabels(['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'UCEC'])
ax1.set_yticklabels(['LR', 'RF', 'Net', 'ATGC'])

vmax = np.max(np.max(context_recall_matrix_normed, axis=-1))
vmin = np.min(np.min(context_recall_matrix_normed, axis=-1))
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})
sns.heatmap(context_recall_matrix_normed, annot=np.around(context_recall_matrix * 100).astype(np.int32), vmin=vmin, vmax=vmax, cmap=myblue, ax=ax2, cbar=False, fmt='d', annot_kws={'fontsize': 10}, square=True)
ax2.tick_params(axis='x', length=0, width=0, labelsize=8, rotation=270)
ax2.tick_params(axis='y', length=0, width=0, labelsize=8, rotation=0)
ax2.set_title('Context Recalls')
ax2.set_yticks(np.array(range(4)) + .5)
ax2.set_xticks(np.array(range(24)) + .5)
ax2.set_xticklabels(['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'UCEC'])
ax2.set_yticklabels(['LR', 'RF', 'Net', 'ATGC'])
plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'context_metrics.png', dpi=600)


gene_precision_matrix_normed = gene_precision_matrix / np.sum(gene_precision_matrix, axis=0)
gene_recall_matrix_normed = gene_recall_matrix / np.sum(gene_recall_matrix, axis=0)

fig = plt.figure()
fig.subplots_adjust(left=.07,
                    right=.99,
                    )
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
vmax = np.max(np.max(gene_precision_matrix_normed, axis=-1))
vmin = np.min(np.min(gene_precision_matrix_normed, axis=-1))
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})
sns.heatmap(gene_precision_matrix_normed, annot=np.around(gene_precision_matrix * 100).astype(np.int32), vmin=vmin, vmax=vmax, cmap=myblue, ax=ax1, cbar=False, fmt='d', annot_kws={'fontsize': 10}, square=True)
ax1.tick_params(axis='x', length=0, width=0, labelsize=8, rotation=270)
ax1.tick_params(axis='y', length=0, width=0, labelsize=8, rotation=0)
ax1.set_title('Gene Precisions')
ax1.set_yticks(np.array(range(4)) + .5)
ax1.set_xticks(np.array(range(24)) + .5)
ax1.set_xticklabels(['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'UCEC'])
ax1.set_yticklabels(['LR', 'RF', 'Net', 'ATGC'])

vmax = np.max(np.max(gene_recall_matrix_normed, axis=-1))
vmin = np.min(np.min(gene_recall_matrix_normed, axis=-1))
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})
sns.heatmap(gene_recall_matrix_normed, annot=np.around(gene_recall_matrix * 100).astype(np.int32), vmin=vmin, vmax=vmax, cmap=myblue, ax=ax2, cbar=False, fmt='d', annot_kws={'fontsize': 10}, square=True)
ax2.tick_params(axis='x', length=0, width=0, labelsize=8, rotation=270)
ax2.tick_params(axis='y', length=0, width=0, labelsize=8, rotation=0)
ax2.set_title('Gene Recalls')
ax2.set_yticks(np.array(range(4)) + .5)
ax2.set_xticks(np.array(range(24)) + .5)
ax2.set_xticklabels(['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'UCEC'])
ax2.set_yticklabels(['LR', 'RF', 'Net', 'ATGC'])
plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'gene_metrics.png', dpi=600)
