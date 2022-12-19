import numpy as np
from scipy.stats import entropy
from model.Sample_MIL import RaggedModels, InstanceModels
import tensorflow as tf
from model import DatasetsUtils
import pickle
import pathlib
import pylab as plt


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
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
class_counts = dict(samples['type'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] > 125]
samples = samples.loc[samples['type'].isin(labels_to_use)]

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

indexes = [np.where(D['sample_idx'] == idx) for idx in samples.index]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')
dropout = .4
index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=.4)
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

test_idx, weights = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'context_weights.pkl', 'rb'))
sequence_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 16, 16], fusion_dimension=128)
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=y_label.shape[-1], output_dims=[y_label.shape[-1]], mil_hidden=[256], attention_layers=[], dropout=.5, instance_dropout=.5, regularization=0, input_dropout=.4)
mil.model.set_weights(weights[0])

idx_test = test_idx[0]
ds_test = tf.data.Dataset.from_tensor_slices(((
                                               five_p_loader_eval(idx_test),
                                               three_p_loader_eval(idx_test),
                                               ref_loader_eval(idx_test),
                                               alt_loader_eval(idx_test),
                                               strand_loader_eval(idx_test),
                                           ),
                                            (
                                                tf.gather(y_label, idx_test),
                                            ),
                                            tf.gather(y_weights, idx_test)
                                            ))

ds_test = ds_test.batch(500, drop_remainder=False)
attention = mil.attention_model.predict(ds_test).numpy()
cancer_to_code = {cancer: index for index, cancer in enumerate(A.cat.categories)}
instances = mil.hidden_model.predict(ds_test).numpy()
aggregations = mil.aggregation_model.predict(ds_test)
prediction_probabilities = mil.model.predict(ds_test)
z = np.exp(prediction_probabilities - np.max(prediction_probabilities, axis=1, keepdims=True))
prediction_probabilities = z / np.sum(z, axis=1, keepdims=True)

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


####tumor heatmap
ramped_average_aggregations = []
accuracies = []
cutoffs = []
for cancer in cancer_to_code:
    cutoffs.append(np.percentile(np.concatenate([i[:, cancer_to_code[cancer]] for i in attention]), 95))

for cancer in cancer_to_code:
    mask = samples.iloc[idx_test]['type'] == cancer
    ramped_weighted_sums = []
    for sample_instances, sample_attention in zip(instances[mask], attention[mask]):
        temp = []
        for head in cancer_to_code:
            attention_mask = sample_attention[:, cancer_to_code[head]] > cutoffs[cancer_to_code[head]]
            temp.append(np.sum(sample_instances[attention_mask] * sample_attention[attention_mask, cancer_to_code[head], np.newaxis], axis=0))
        ramped_weighted_sums.append(np.array(temp))
    ramped_average_aggregations.append(np.mean(np.array(ramped_weighted_sums), axis=0))
    accuracies.append(sum(np.argmax(y_label[idx_test][mask], axis=-1) == np.argmax(prediction_probabilities[mask], axis=-1)) / sum(mask))

filtered_aggregations = []
for i in ramped_average_aggregations:
    z = np.exp(i.T - np.max(i.T, axis=1, keepdims=True))
    probabilities = z / np.sum(z, axis=1, keepdims=True)
    entropies = entropy(probabilities, axis=-1)
    matrix = i.T[entropies < np.percentile(entropies, 5)]
    filtered_aggregations.append(matrix / matrix.sum(axis=-1, keepdims=True))

vmax = np.percentile(np.concatenate([i.flatten() for i in filtered_aggregations]), 99)
myblue = make_colormap({0: '#ffffff',
                        vmax * .5: '#91a8ee',
                        vmax: '#4169E1'})

fig = plt.figure()
fig.subplots_adjust(hspace=0,
                    left=.09,
                    right=.99,
                    bottom=.13,
                    top=.99)
gs = fig.add_gridspec(24, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[4, 0])
ax6 = fig.add_subplot(gs[5, 0])
ax7 = fig.add_subplot(gs[6, 0])
ax8 = fig.add_subplot(gs[7, 0])
ax9 = fig.add_subplot(gs[8, 0])
ax10 = fig.add_subplot(gs[9, 0])
ax11 = fig.add_subplot(gs[10, 0])
ax12 = fig.add_subplot(gs[11, 0])
ax13 = fig.add_subplot(gs[12, 0])
ax14 = fig.add_subplot(gs[13, 0])
ax15 = fig.add_subplot(gs[14, 0])
ax16 = fig.add_subplot(gs[15, 0])
ax17 = fig.add_subplot(gs[16, 0])
ax18 = fig.add_subplot(gs[17, 0])
ax19 = fig.add_subplot(gs[18, 0])
ax20 = fig.add_subplot(gs[19, 0])
ax21 = fig.add_subplot(gs[20, 0])
ax22 = fig.add_subplot(gs[21, 0])
ax23 = fig.add_subplot(gs[22, 0])
ax24 = fig.add_subplot(gs[23, 0])

for matrix, ax in zip(filtered_aggregations, [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24]):
    ax.imshow(matrix,
               cmap=myblue,
               vmin=0,
               vmax=vmax,
               aspect='auto',
              interpolation='nearest')

for cancer, ax in zip(cancer_to_code, [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24]):
    ax.tick_params(length=0, width=0, labelsize=8)
    ax.set_yticks([3.5])
    ax.set_yticklabels([cancer])
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23]:
    ax.set_xticks([])
ax24.set_xticks(np.array((range(len(cancer_to_code)))))
ax24.set_xticklabels(list(cancer_to_code.keys()), rotation=270)
ax24.set_xlabel('Average Ramped Sums', fontsize=14)
ax18.set_title('Attention Head', fontsize=14, rotation=90, position=[-.08, 0])

plt.savefig(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'figures' / 'tumor_figure.png', dpi=600)
