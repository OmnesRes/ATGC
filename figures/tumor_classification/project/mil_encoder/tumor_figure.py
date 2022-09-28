import numpy as np
import pandas as pd
from scipy.stats import entropy
import re
from Bio.Seq import Seq
from model.Sample_MIL import RaggedModels, InstanceModels
import tensorflow as tf
from model import DatasetsUtils
import pickle
import pathlib
import logomaker as lm
import pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage


path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
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
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=y_label.shape[-1], output_types=['other'], mil_hidden=[256], attention_layers=[], dropout=.5, instance_dropout=.5, regularization=0, input_dropout=.4)
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

ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
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
    accuracies.append(sum(np.argmax(y_label[idx_test][mask], axis=-1) == np.argmax(probabilities[mask], axis=-1)) / sum(mask))


flattened_aggregations = np.concatenate(ramped_average_aggregations, axis=-1).T
z = np.exp(flattened_aggregations - np.max(flattened_aggregations, axis=1, keepdims=True))
probabilities = z / np.sum(z, axis=1, keepdims=True)
entropies = entropy(probabilities, axis=-1)
flattened_aggregations = flattened_aggregations[entropies < np.percentile(entropies, 5)]

matrix = flattened_aggregations / flattened_aggregations.sum(axis=-1, keepdims=True)
# matrix = flattened_aggregations
vmax = np.percentile(matrix, 99)
myblue = make_colormap({0: '#ffffff', vmax * .01: '#e9eefc', vmax * .5: '#91a8ee', vmax: '#4169E1'})

fig = plt.figure()
ax = fig.add_subplot(111)
figure_matrix = ax.imshow(matrix,
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')


code_to_cancer = {j: i for i, j in zip(cancer_to_code.keys(), cancer_to_code.values())}
for i in np.argsort(accuracies)[::-1]:
    print(code_to_cancer[i])


##sample heatmap
mask = samples.iloc[idx_test]['type'] == 'SKCM'
aggregations = aggregation[:, cancer_to_code['SKCM'], :][mask]
prediction = probabilities[mask][:, cancer_to_code['SKCM']]

matrix = np.array(aggregations)[np.argsort(prediction)[::-1]].T
mask = np.sum(matrix, axis=-1) == 0
matrix = matrix[~mask]

vmax = np.percentile(matrix / matrix.sum(axis=0, keepdims=True), 99)
myblue = make_colormap({0: '#ffffff', vmax * .01: '#e9eefc', vmax * .5:'#91a8ee', vmax: '#4169E1'})

fig = plt.figure()
gs = fig.add_gridspec(2, 1, height_ratios=[15, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
figure_matrix = ax1.imshow(matrix / matrix.sum(axis=0, keepdims=True),
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

vmax = 1
myblue = make_colormap({0: '#ffffff', vmax: '#4169E1'})
prediction_matrix = ax2.imshow(prediction[np.argsort(prediction)[::-1]][np.newaxis, :],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                          interpolation='nearest')

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

##instance features
mask = samples.iloc[idx_test]['type'] == 'SKCM'
prediction = prediction_probabilities[mask][:, cancer_to_code['SKCM']]
instances = mil.hidden_model.predict(ds_test).numpy()[mask][np.argsort(prediction)[::-1][0]]
skcm_instance_attention_skcm = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['SKCM']]
skcm_instance_attention_skcm = skcm_instance_attention_skcm - min(skcm_instance_attention_skcm)
lusc_instance_attention_skcm = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['LUSC']]
lusc_instance_attention_skcm = lusc_instance_attention_skcm - min(lusc_instance_attention_skcm)
luad_instance_attention_skcm = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['LUAD']]
luad_instance_attention_skcm = luad_instance_attention_skcm - min(luad_instance_attention_skcm)
skcm_matrix = instances.T
label = np.repeat(0, len(instances))

mask = samples.iloc[idx_test]['type'] == 'LUSC'
prediction = prediction_probabilities[mask][:, cancer_to_code['LUSC']]
instances = mil.hidden_model.predict(ds_test).numpy()[mask][np.argsort(prediction)[::-1][0]]
skcm_instance_attention_lusc = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['SKCM']]
skcm_instance_attention_lusc = skcm_instance_attention_lusc - min(skcm_instance_attention_lusc)
lusc_instance_attention_lusc = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['LUSC']]
lusc_instance_attention_lusc = lusc_instance_attention_lusc - min(lusc_instance_attention_lusc)
luad_instance_attention_lusc = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['LUAD']]
luad_instance_attention_lusc = luad_instance_attention_lusc - min(luad_instance_attention_lusc)
lusc_matrix = instances.T
label = np.concatenate([label, np.repeat(1, len(instances))])


mask = samples.iloc[idx_test]['type'] == 'LUAD'
prediction = prediction_probabilities[mask][:, cancer_to_code['LUAD']]
instances = mil.hidden_model.predict(ds_test).numpy()[mask][np.argsort(prediction)[::-1][0]]
skcm_instance_attention_luad = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['SKCM']]
skcm_instance_attention_luad = skcm_instance_attention_luad - min(skcm_instance_attention_luad)
lusc_instance_attention_luad = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['LUSC']]
lusc_instance_attention_luad = lusc_instance_attention_luad - min(lusc_instance_attention_luad)
luad_instance_attention_luad = attention[mask][np.argsort(prediction)[::-1][0]][:, cancer_to_code['LUAD']]
luad_instance_attention_luad = luad_instance_attention_luad - min(luad_instance_attention_luad)
luad_matrix = instances.T
label = np.concatenate([label, np.repeat(2, len(instances))])


skcm_instance_attention = np.concatenate([skcm_instance_attention_skcm, skcm_instance_attention_lusc, skcm_instance_attention_luad])
lusc_instance_attention = np.concatenate([lusc_instance_attention_skcm, lusc_instance_attention_lusc, lusc_instance_attention_luad])
luad_instance_attention = np.concatenate([luad_instance_attention_skcm, luad_instance_attention_lusc, luad_instance_attention_luad])

# cutoff = np.percentile(skcm_instance_attention, 90)
# instance_attention = instance_attention * (instance_attention > cutoff).astype(int)
matrix = np.concatenate([skcm_matrix, lusc_matrix, luad_matrix], axis=-1)
mask = np.sum(matrix, axis=-1) == 0
matrix = matrix[~mask]


Z = linkage(matrix.T, 'ward')
dn=dendrogram(Z, leaf_rotation=90, leaf_font_size=8, color_threshold=1)
matrix=matrix.T[list(dn.values())[3]].T


vmax = np.percentile(matrix / matrix.sum(axis=-1, keepdims=True), 99)
# vmax= max(np.max(matrix, axis=-1))
myblue = make_colormap({0: '#ffffff', vmax: '#4169E1'})

fig = plt.figure()
fig.subplots_adjust(hspace=0)
gs = fig.add_gridspec(5, 1, height_ratios=[15, 1, 1, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[4, 0])
figure_matrix = ax1.imshow(matrix / matrix.sum(axis=-1, keepdims=True),
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

vmin = min(skcm_instance_attention)
vmax = np.percentile(skcm_instance_attention, 98)
myblue = make_colormap({vmin: '#ffffff', vmax * .6: '#e9eefc', vmax: '#4169E1'})
prediction_matrix = ax2.imshow(skcm_instance_attention[list(dn.values())[3]][np.newaxis, :],
                           cmap=myblue,
                           vmin=vmin,
                           vmax=vmax,
                            aspect='auto',
                          interpolation='nearest')

vmin = min(lusc_instance_attention)
vmax = np.percentile(lusc_instance_attention, 98)
myblue = make_colormap({vmin: '#ffffff', vmax * .6: '#e9eefc', vmax: '#4169E1'})
prediction_matrix = ax3.imshow(lusc_instance_attention[list(dn.values())[3]][np.newaxis, :],
                           cmap=myblue,
                           vmin=vmin,
                           vmax=vmax,
                            aspect='auto',
                          interpolation='nearest')


vmin = min(luad_instance_attention)
vmax = np.percentile(luad_instance_attention, 98)
myblue = make_colormap({vmin: '#ffffff', vmax * .6: '#e9eefc', vmax: '#4169E1'})
prediction_matrix = ax4.imshow(luad_instance_attention[list(dn.values())[3]][np.newaxis, :],
                           cmap=myblue,
                           vmin=vmin,
                           vmax=vmax,
                            aspect='auto',
                          interpolation='nearest')


label_matrix = ax5.imshow(label[list(dn.values())[3]][np.newaxis, :],
                           # cmap=myblue,
                           # vmin=vmin,
                           # vmax=vmax,
                            aspect='auto',
                          interpolation='nearest')

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])




import seaborn as sns
sns.clustermap(matrix, row_cluster=False)

# test_indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].index]
# refs = tcga_maf['Reference_Allele'].values[np.concatenate([i[0] for i in test_indexes])]
# alts = tcga_maf['Tumor_Seq_Allele2'].values[np.concatenate([i[0] for i in test_indexes])]
# five_ps = tcga_maf['five_p'].values[np.concatenate([i[0] for i in test_indexes])]
# three_ps = tcga_maf['three_p'].values[np.concatenate([i[0] for i in test_indexes])]
