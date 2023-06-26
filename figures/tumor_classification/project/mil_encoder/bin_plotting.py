import numpy as np
from model.Sample_MIL import RaggedModels, InstanceModels
import tensorflow as tf
from model import DatasetsUtils
import pickle
import pathlib
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-2], True)
tf.config.experimental.set_visible_devices(physical_devices[-2], 'GPU')

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'data' / 'data.pkl', 'rb'))
tcga_maf['Hugo_Symbol'] = tcga_maf['Hugo_Symbol'].astype('category')
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
class_counts = dict(samples['type'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] > 125]
samples = samples.loc[samples['type'].isin(labels_to_use)]

bin_size = 3095677412 // 3000

tcga_maf['bin'] = tcga_maf['genome_position'].values // bin_size
D['bin'] = np.concatenate(tcga_maf[['bin']].astype('category').apply(lambda x: x.cat.codes).values + 1)
code_to_bin = {i: j for i, j in zip(D['bin'], tcga_maf['bin'].values)}
bin_to_code = {i: j for i, j in zip(tcga_maf['bin'].values, D['bin'])}

chromosomes = {}
for i in list(range(1, 23))+['X', 'Y']:
    with open(cwd / 'files' / 'chromosomes' / ('chr' + str(i) + '.txt')) as f:
        chromosomes[str(i)] = f.read()

temp_pos = 0
chr_ends = [0]
for i in list(range(1, 23))+['X']:
    temp_pos += len(chromosomes[str(i)])
    chr_ends.append(bin_to_code[temp_pos // bin_size])

input_dim = max(D['bin'])
dropout = 0

A = samples['type'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot
cancer_to_code = {cancer: index for index, cancer in enumerate(A.cat.categories)}


bin_encoder = InstanceModels.GeneEmbed(shape=(), input_dim=input_dim, dim=128)
mil = RaggedModels.MIL(instance_encoders=[bin_encoder.model], sample_encoders=[], heads=y_label.shape[-1], output_dims=[y_label.shape[-1]], mil_hidden=[], attention_layers=[], regularization=0, input_dropout=dropout, weight_decay=.0005)
test_idx, weights = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'bin_embed_weights.pkl', 'rb'))
weight_matrix = weights[0][0]
sample = np.array([np.arange(1, max(code_to_bin.keys()) + 1)], dtype='object')
sample_loader = DatasetsUtils.Map.FromNumpy(sample, tf.int16, dropout=0)
ds_sample = tf.data.Dataset.from_tensor_slices(((
                                           sample_loader([0]),
                                       ),
                                        (
                                        ),
                                        )).batch(1)

attention_folds = []
for i in range(5):
    mil.model.set_weights(weights[i])
    attention_folds.append(mil.attention_model.predict(ds_sample).numpy())
weight_matrix = weights[0][0]

Z = linkage(weight_matrix.T, 'ward')
dn = dendrogram(Z, leaf_rotation=90, leaf_font_size=8, color_threshold=1.2)
plt.show()

weight_matrix = weight_matrix.T[list(dn.values())[3]].T

position_z_scores = []
for attention in attention_folds:
    fold_z_scores = []
    for head in cancer_to_code.values():
        fold_z_scores.append((attention[0, :, head] - np.mean(attention[0, :, head])) / np.std(attention[0, :, head], ddof=1))
    position_z_scores.append(np.mean(fold_z_scores, axis=0))

position_z_scores = np.mean(position_z_scores, axis=0)

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


fig = plt.figure()
fig.subplots_adjust(left=.07,
                    bottom=.08,
                    right=.993,
                    top=.6,
                    hspace=.05)
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax1.fill_between([0, max(code_to_bin.keys())], -1, 1, color='k', alpha=.2, edgecolor='none')
ax1.bar(np.arange(0, max(code_to_bin.keys())), position_z_scores,
        linewidth=.5,
        edgecolor='#1f77b4')

vmin = np.percentile(weight_matrix, 50)
vmax = np.percentile(weight_matrix, 99.9)
myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})

weight_heatmap = ax2.imshow(weight_matrix.T,
                           cmap=myblue,
                           vmin=vmin,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

ax1.tick_params(width=0, length=0, labelsize=8)
ax2.tick_params(width=0, length=0)
ax1.set_xticks([])
ax1.set_ylabel('Attention Z-scores')
ax2.set_xticks(chr_ends)
ax2.set_xticklabels(list(range(1, 23))+['X', 'Y'], fontsize=6, rotation=-45)
ax2.set_yticks([])
ax2.set_ylabel('Features', fontsize=10)
ax2.set_xlabel('Chromosome')
ax1.set_xlim(0, max(code_to_bin.keys()) - 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
plt.savefig('test.png', dpi=600)
# plt.show()
# plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'chromosome_attention.png', dpi=600)


##scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=.15,
                    bottom=.10,
                    right=1,
                    top=.99)
bin_counts = dict(zip(*np.unique(tcga_maf['bin'].values, return_counts=True)))
ax.scatter(position_z_scores, np.log([bin_counts[code_to_bin[i+1]] for i in range(max(code_to_bin.keys()))]), s=20, alpha=.4, edgecolor='none')
ax.set_yticklabels([1, 10, 100, 1000, 10000])
ax.set_yticks([np.log(i) for i in [1, 10, 100, 1000, 10000]])
ax.set_ylabel('Mutations per Bin', fontsize=16)
ax.set_xlabel('Attention Z-score', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds([0, np.log(10000)])
ax.spines['bottom'].set_bounds([-3, 4])
plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'bin_size_scatter.png', dpi=600)
# plt.show()


##correlation with gene attention

D['genes'] = np.concatenate(tcga_maf[['Hugo_Symbol']].apply(lambda x: x.cat.codes).values + 1)
code_to_gene = {i: j for i, j in zip(D['genes'], tcga_maf['Hugo_Symbol'].values)}
gene_to_bin = {}
for i in tcga_maf.itertuples():
    gene_to_bin[i.Hugo_Symbol] = gene_to_bin.get(i.Hugo_Symbol, []) + [i.bin]
for i in gene_to_bin:
    gene_to_bin[i] = set(gene_to_bin[i])


input_dim = max(D['genes'])
dropout = 0

gene_encoder = InstanceModels.GeneEmbed(shape=(), input_dim=input_dim, dim=128)
mil = RaggedModels.MIL(instance_encoders=[gene_encoder.model], sample_encoders=[], heads=y_label.shape[-1], output_dims=[y_label.shape[-1]], mil_hidden=[], attention_layers=[], instance_dropout=0, regularization=0, input_dropout=dropout, weight_decay=.0005)
test_idx, weights = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'gene_weights2.pkl', 'rb'))
weight_matrix = weights[0][0]
sample = np.array([np.arange(1, max(D['genes']) + 1)], dtype='object')
sample_loader = DatasetsUtils.Map.FromNumpy(sample, tf.int16, dropout=0)

ds_sample = tf.data.Dataset.from_tensor_slices(((
                                           sample_loader([0]),
                                       ),
                                        (
                                        ),
                                        )).batch(1)

attention_folds = []
for i in range(5):
    mil.model.set_weights(weights[i])
    attention_folds.append(mil.attention_model.predict(ds_sample).numpy())

gene_z_scores = []
for attention in attention_folds:
    fold_z_scores = []
    for head in cancer_to_code.values():
        fold_z_scores.append((attention[0, :, head] - np.mean(attention[0, :, head])) / np.std(attention[0, :, head], ddof=1))
    gene_z_scores.append(np.mean(fold_z_scores, axis=0))

gene_z_scores = np.mean(gene_z_scores, axis=0)
##match up genes with their bins
mapped_values = []
for index, i in enumerate(gene_z_scores):
    temp_values = []
    for bin in gene_to_bin[code_to_gene[index + 1]]:
        temp_values.append(position_z_scores[bin_to_code[bin] - 1])
    mapped_values.append(np.mean(temp_values))


fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=.1,
                    bottom=.10,
                    right=.97,
                    top=.99)
bin_counts = dict(zip(*np.unique(tcga_maf['bin'].values, return_counts=True)))
ax.scatter(gene_z_scores, mapped_values, s=10, alpha=.3, edgecolor='none')
ax.set_xticks([-2, 0, 2, 4, 6, 8, 10])
ax.set_ylabel('Bin Z-scores', fontsize=16)
ax.set_xlabel('Gene Z-scores', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds([-3, 4])
ax.spines['bottom'].set_bounds([-2, 10])
plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'bin_gene_attention.png', dpi=600)
# plt.show()

##investigate top genes
top_genes = np.argsort(gene_z_scores)[-5:]
[code_to_gene[i + 1] for i in top_genes]
#['VHL', 'KRAS', 'BRAF', 'APC', 'IDH1']
