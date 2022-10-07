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


abbreviations = {'Muscle-Invasive Bladder Carcinoma': 'MIBC',
                 'Infiltrating Ductal Breast Carcinoma': 'IDBC',
                 'Invasive Lobular Breast Carcinoma': 'ILBC',
                'Cervical Squamous Cell Carcinoma': 'CSCC',
                 'Colorectal Adenocarcinoma': 'COAD',
                'Glioblastoma': 'GBM',
                 'Head and Neck Squamous Cell Carcinoma': 'HNSC',
                 'Clear Cell Renal Cell Carcinoma': 'KIRC',
                 'Papillary Renal Cell Carcinoma': 'KIRP',
                 'Astrocytoma': 'AC',
                 'Oligoastrocytoma': 'OAC',
                 'Oligodendroglioma': 'ODG',
                 'Hepatocellular Carcinoma': 'LIHC',
                 'Lung Adenocarcinoma': 'LUAD',
                 'Lung Squamous Cell Carcinoma': 'LUSC',
                 'Ovarian Serous Adenocarcinoma': 'OV',
                 'Adenocarcinoma, Pancreas': 'PAAD',
                 'PCPG': 'PCPG',
                 'Prostate Acinar Adenocarcinoma': 'PRAD',
                 'SARC': 'SARC',
                 'Cutaneous Melanoma': 'SKCM',
                 'Gastric Adenocarcinoma': 'STAD',
                 'TGCT': 'TGCT',
                 'Thyroid Gland Follicular Carcinoma': 'TGFC',
                 'Thyroid Gland Papillary Carcinoma': 'TGPC',
                 'Endometrial Endometrioid Adenocarcinoma': 'EEA',
                 'Endometrial Serous Adenocarcinoma': 'ESA'}


D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'data' / 'data.pkl', 'rb'))

samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'PCPG' if x == 'Paraganglioma' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'PCPG' if x == 'Pheochromocytoma' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'SARC' if x == 'Desmoid-Type Fibromatosis' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'SARC' if x == 'Leiomyosarcoma' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'SARC' if x == 'Liposarcoma' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'SARC' if x == 'Malignant Peripheral Nerve Sheath Tumor' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'SARC' if x == 'Myxofibrosarcoma' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'SARC' if x == 'Synovial Sarcoma' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'SARC' if x == 'Undifferentiated Pleomorphic Sarcoma' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'TGCT' if x == 'Testicular Non-Seminomatous Germ Cell Tumor' else x)
samples['NCIt_label'] = samples['NCIt_label'].apply(lambda x: 'TGCT' if x == 'Testicular Seminoma' else x)

labels_to_use = ['Muscle-Invasive Bladder Carcinoma', 'Infiltrating Ductal Breast Carcinoma',
                 'Invasive Lobular Breast Carcinoma', 'Cervical Squamous Cell Carcinoma',
                 'Colorectal Adenocarcinoma', 'Glioblastoma', 'Head and Neck Squamous Cell Carcinoma',
                 'Clear Cell Renal Cell Carcinoma', 'Papillary Renal Cell Carcinoma',
                 'Astrocytoma', 'Oligoastrocytoma', 'Oligodendroglioma', 'Hepatocellular Carcinoma',
                 'Lung Adenocarcinoma', 'Lung Squamous Cell Carcinoma', 'Ovarian Serous Adenocarcinoma',
                 'Adenocarcinoma, Pancreas', 'PCPG', 'Prostate Acinar Adenocarcinoma',
                 'SARC', 'Cutaneous Melanoma', 'Gastric Adenocarcinoma',
                 'TGCT', 'Thyroid Gland Follicular Carcinoma', 'Thyroid Gland Papillary Carcinoma',
                 'Endometrial Endometrioid Adenocarcinoma', 'Endometrial Serous Adenocarcinoma']

samples = samples.loc[samples['NCIt_label'].isin(labels_to_use)]

A = samples['NCIt_label'].astype('category')
cancer_to_code = {cancer: index for index, cancer in enumerate(A.cat.categories)}

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

context_mil_precisions, context_mil_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'mil_encoder' / 'results' / 'context_metrics.pkl', 'rb'))
gene_mil_precisions, gene_mil_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'mil_encoder' / 'results' / 'gene_metrics.pkl', 'rb'))
context_net_precisions, context_net_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'neural_net' / 'results' / 'context_metrics.pkl', 'rb'))
gene_net_precisions, gene_net_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'neural_net' / 'results' / 'gene_metrics.pkl', 'rb'))
context_forest_precisions, context_forest_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'standard' / 'results' / 'context_forest_metrics.pkl', 'rb'))
gene_forest_precisions, gene_forest_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'standard' / 'results' / 'gene_forest_metrics.pkl', 'rb'))
context_logistic_precisions, context_logistic_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'standard' / 'results' / 'context_logistic_metrics.pkl', 'rb'))
gene_logistic_precisions, gene_logistic_recalls =pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'standard' / 'results' / 'gene_logistic_metrics.pkl', 'rb'))

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
ax1.set_xticks(np.array(range(27)) + .5)
ax1.set_xticklabels([abbreviations[i] for i in cancer_to_code])
ax1.set_yticklabels(['LR', 'RF', 'Net', 'ATGC'])

vmax = np.max(np.max(context_recall_matrix_normed, axis=-1))
vmin = np.min(np.min(context_recall_matrix_normed, axis=-1))
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})
sns.heatmap(context_recall_matrix_normed, annot=np.around(context_recall_matrix * 100).astype(np.int32), vmin=vmin, vmax=vmax, cmap=myblue, ax=ax2, cbar=False, fmt='d', annot_kws={'fontsize': 10}, square=True)
ax2.tick_params(axis='x', length=0, width=0, labelsize=8, rotation=270)
ax2.tick_params(axis='y', length=0, width=0, labelsize=8, rotation=0)
ax2.set_title('Context Recalls')
ax2.set_yticks(np.array(range(4)) + .5)
ax2.set_xticks(np.array(range(27)) + .5)
ax2.set_xticklabels([abbreviations[i] for i in cancer_to_code])
ax2.set_yticklabels(['LR', 'RF', 'Net', 'ATGC'])
# plt.savefig('context_metrics.png', dpi=600)


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
ax1.set_xticks(np.array(range(27)) + .5)
ax1.set_xticklabels([abbreviations[i] for i in cancer_to_code])
ax1.set_yticklabels(['LR', 'RF', 'Net', 'ATGC'])

vmax = np.max(np.max(gene_recall_matrix_normed, axis=-1))
vmin = np.min(np.min(gene_recall_matrix_normed, axis=-1))
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})
sns.heatmap(gene_recall_matrix_normed, annot=np.around(gene_recall_matrix * 100).astype(np.int32), vmin=vmin, vmax=vmax, cmap=myblue, ax=ax2, cbar=False, fmt='d', annot_kws={'fontsize': 10}, square=True)
ax2.tick_params(axis='x', length=0, width=0, labelsize=8, rotation=270)
ax2.tick_params(axis='y', length=0, width=0, labelsize=8, rotation=0)
ax2.set_title('Gene Recalls')
ax2.set_yticks(np.array(range(4)) + .5)
ax2.set_xticks(np.array(range(27)) + .5)
ax2.set_xticklabels([abbreviations[i] for i in cancer_to_code])
ax2.set_yticklabels(['LR', 'RF', 'Net', 'ATGC'])
# plt.savefig('gene_metrics.png', dpi=600)