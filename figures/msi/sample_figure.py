import numpy as np
from scipy.stats import entropy
from model.Sample_MIL import RaggedModels, InstanceModels
import tensorflow as tf
from model import DatasetsUtils
import pickle
import pathlib
import pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))
del tcga_maf
fold = 4

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]
indexes = [np.where(D['sample_idx'] == idx) for idx in samples.index]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')
dropout = .4
index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=dropout)
five_p_loader = DatasetsUtils.Map.FromNumpyandIndices(five_p, tf.int16)
three_p_loader = DatasetsUtils.Map.FromNumpyandIndices(three_p, tf.int16)
ref_loader = DatasetsUtils.Map.FromNumpyandIndices(ref, tf.int16)
alt_loader = DatasetsUtils.Map.FromNumpyandIndices(alt, tf.int16)
strand_loader = DatasetsUtils.Map.FromNumpyandIndices(strand, tf.float32)

five_p_loader_eval = DatasetsUtils.Map.FromNumpy(five_p, tf.int16)
three_p_loader_eval = DatasetsUtils.Map.FromNumpy(three_p, tf.int16)
ref_loader_eval = DatasetsUtils.Map.FromNumpy(ref, tf.int16)
alt_loader_eval = DatasetsUtils.Map.FromNumpy(alt, tf.int16)
strand_loader_eval = DatasetsUtils.Map.FromNumpy(strand, tf.float32)

A = samples.msi_status.astype('category')
classes = A.cat.categories.values

# set y label and weights
y_label = A.cat.codes.values[:, np.newaxis]
cancer_labels = [i if i in ['STAD', 'UCEC', 'COAD'] else 'other' for i in samples['type']]
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(cancer_labels, y_label[:, 0])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(cancer_labels, y_label[:, 0])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)

test_idx, weights = pickle.load(open(cwd / 'figures' / 'msi' / 'results' / 'run.pkl', 'rb'))

sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=128)
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=y_label.shape[-1], output_types=['other'], mil_hidden=(256, 128), attention_layers=[], dropout=.5, instance_dropout=.5, regularization=.2, input_dropout=dropout)

idx_test = test_idx[fold]
mil.model.set_weights(weights[fold])
ds_test = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(idx_test),
                                       three_p_loader_eval(idx_test),
                                       ref_loader_eval(idx_test),
                                       alt_loader_eval(idx_test),
                                       strand_loader_eval(idx_test),
                                    ),
                                   tf.gather(y_label, idx_test),
                                   ))
ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
attention = mil.attention_model.predict(ds_test).numpy()
instances = mil.hidden_model.predict(ds_test).numpy()
aggregations = mil.aggregation_model.predict(ds_test)
prediction_probabilities = tf.nn.sigmoid(mil.model.predict(ds_test))

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


##sample heatmap
label = (samples.iloc[idx_test]['msi_status'] == 'high').astype(np.int32).values
predictions = 1 - prediction_probabilities[:, 0].numpy()
matrix = aggregations[:, 0, :][np.argsort(predictions)[::-1]].T
z = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
probabilities = z / np.sum(z, axis=1, keepdims=True)
entropies = entropy(probabilities, axis=-1)
matrix = matrix[entropies < np.percentile(entropies, 90)]

Z = linkage(matrix, 'ward')
dn = dendrogram(Z, leaf_rotation=90, leaf_font_size=8, color_threshold=1)
matrix = matrix[list(dn.values())[3]]

vmax = np.percentile(matrix / np.sum(matrix, axis=-1, keepdims=True), 99)
myblue = make_colormap({0: '#ffffff',
                        vmax * .01: '#e9eefc',
                        vmax * .5: '#91a8ee',
                        vmax: '#4169E1'})

fig = plt.figure()
fig.subplots_adjust(hspace=0,
                    left=.05,
                    right=.99,
                    bottom=.03,
                    top=.94)
gs = fig.add_gridspec(5, 1, height_ratios=[25, 0, 1, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], zorder=-1000)
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[3, 0], zorder=-1000)
ax5 = fig.add_subplot(gs[4, 0])

figure_matrix = ax1.imshow(matrix / np.sum(matrix, axis=-1, keepdims=True),
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

vmax = 1
mycat = make_colormap({0: '#ffffff', vmax: '#000000'})
prediction_matrix = ax5.imshow(label[np.argsort(predictions)[::-1]][np.newaxis, :],
                           cmap=mycat,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                           interpolation='nearest')

vmax = 1
myblue = make_colormap({0: '#ffffff', vmax: '#4169E1'})
prediction_matrix = ax3.imshow(predictions[np.argsort(predictions)[::-1]][np.newaxis, :],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                           interpolation='nearest')

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_xticks([])
    ax.set_yticks([])
for ax in [ax2, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

ax1.set_ylabel('Features', fontsize=16)
ax1.set_xlabel('Samples', fontsize=16)
ax1.xaxis.set_label_position('top')
ax5.set_xlabel('MSI Status', fontsize=10, labelpad=2)
ax3.set_xlabel('Model Probability', fontsize=10, labelpad=2)
plt.savefig(cwd / 'figures' / 'msi' / 'sample_figure.png', dpi=600)
