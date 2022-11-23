import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from model.Sample_MIL import RaggedModels, InstanceModels
import tensorflow as tf
from model import DatasetsUtils
import pickle
import pathlib
import pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from scipy import spatial

path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys

    sys.path.append(str(cwd))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'data' / 'data.pkl', 'rb'))
fold = 0
tcga_maf['Hugo_Symbol'] = tcga_maf['Hugo_Symbol'].astype('category')
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
class_counts = dict(samples['type'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] > 125]
samples = samples.loc[samples['type'].isin(labels_to_use)]

D['genes'] = np.concatenate(tcga_maf[['Hugo_Symbol']].apply(lambda x: x.cat.codes).values + 1)

input_dim = max(D['genes'])
dropout = .5
indexes = [np.where(D['sample_idx'] == idx) for idx in samples.index]
genes = np.array([D['genes'][i] for i in indexes], dtype='object')
index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=dropout)
genes_loader = DatasetsUtils.Map.FromNumpyandIndices(genes, tf.int16)
genes_loader_eval = DatasetsUtils.Map.FromNumpy(genes, tf.int16, dropout=0)

A = samples['type'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot

y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)
y_weights_loader = DatasetsUtils.Map.FromNumpy(y_weights, tf.float32)

test_idx, weights = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'gene_weights.pkl', 'rb'))
gene_encoder = InstanceModels.GeneEmbed(shape=(), input_dim=input_dim, dim=128)
mil = RaggedModels.MIL(instance_encoders=[gene_encoder.model], sample_encoders=[], heads=y_label.shape[-1], output_dims=[y_label.shape[-1]], mil_hidden=[], attention_layers=[], instance_dropout=0, regularization=0, input_dropout=.5)
mil.model.set_weights(weights[fold])

idx_test = test_idx[fold]
ds_test = tf.data.Dataset.from_tensor_slices(((
                                               genes_loader_eval(idx_test),
                                           ),
                                            (
                                                tf.gather(y_label, idx_test),
                                            ),
                                            tf.gather(y_weights, idx_test)
                                            ))

ds_test = ds_test.batch(len(idx_test), drop_remainder=False)

attention = mil.attention_model.predict(ds_test).numpy()

cancer_to_code = {cancer: index for index, cancer in enumerate(A.cat.categories)}

cancer = 'SKCM'
cancer_attention = np.concatenate([i[:, cancer_to_code[cancer]] for i in attention])
bin = .0002
counts = dict(zip(*np.unique(np.around(cancer_attention, 4), return_counts=True)))
values = [sum([counts.get(np.around((i / 10000) + j / 10000, 4).astype(np.float32), 0) for j in range(int(bin * 10000))]) for i in range(int(min(counts.keys()) * 10000), int(max(counts.keys()) * 10000) + 1, int(bin * 10000))]

##attention histogram
fig = plt.figure()
fig.subplots_adjust(top=1,
bottom=.056,
left=0.032,
right=1)
ax = fig.add_subplot(111)
ax.bar(range(int(min(counts.keys()) * 10000), int(max(counts.keys()) * 10000) + 1, int(bin * 10000)), values, 2, alpha=.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Attention', fontsize=16)
ax.set_ylabel('Variant Density', fontsize=16, labelpad=-5)
plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'figures' / 'gene_attention_skcm.png', dpi=600)


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


##embedding weights
matrix = tf.keras.activations.relu(weights[0][0]).numpy().T
kmeans = KMeans(n_clusters=8, random_state=0).fit(matrix.T)
cluster_similarity = [1 - spatial.distance.cosine(kmeans.cluster_centers_[0], cluster) for cluster in kmeans.cluster_centers_]
cluster_order = np.arange(8)[list(np.argsort(cluster_similarity))[::-1]]
label_dict = {i: j for i, j in zip(cluster_order, np.arange(8))}
labels = np.array([label_dict[i] for i in kmeans.labels_])
ordered_matrix = matrix.T[np.argsort(labels)].T
label = kmeans.labels_[np.argsort(labels)]

Z = linkage(ordered_matrix, 'ward')
dn = dendrogram(Z, leaf_rotation=90, leaf_font_size=8, color_threshold=1)
ordered_matrix = ordered_matrix[list(dn.values())[3]]

fig = plt.figure()
fig.subplots_adjust(left=.05,
                    bottom=.005,
                    right=.99,
                    top=.94,
                    hspace=0)
gs = fig.add_gridspec(2, 1, height_ratios=[25, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

vmax=np.percentile(ordered_matrix, 99)
vmin = 0

myblue = make_colormap({vmin: '#ffffff', vmax: '#4169E1'})

latent_matrix = ax1.imshow(ordered_matrix,
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

myqual = make_colormap({0: '#d53e4f', 1: '#f46d43', 7: '#fdae61', 5: '#000000', 4: '#e6f598', 2: '#abdda4', 6: '#66c2a5', 3: '#3288bd'})

ax2.imshow(label[np.newaxis, :],
           cmap=myqual,
          aspect='auto',
          interpolation='nearest')

for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])
ax1.set_xlabel('Genes', fontsize=16)
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Embedding', fontsize=16)
plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'figures' / 'gene_embedding.png', dpi=600)



##find genes in small cluster
code_to_gene = {index: i for index, i in enumerate(tcga_maf['Hugo_Symbol'].cat.categories)}
gene_codes = np.arange(0, len(kmeans.labels_))[kmeans.labels_ == 5]

for i in [code_to_gene[i] for i in gene_codes]:
    print(i+',')

