import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
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
from sklearn.cluster import KMeans
from scipy import spatial


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


cancer = 'SKCM'
##
test_indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].index]
indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer].index]
refs = tcga_maf['Reference_Allele'].values[np.concatenate([i[0] for i in indexes])]
alts = tcga_maf['Tumor_Seq_Allele2'].values[np.concatenate([i[0] for i in indexes])]
five_ps = tcga_maf['five_p'].values[np.concatenate([i[0] for i in indexes])]
three_ps = tcga_maf['three_p'].values[np.concatenate([i[0] for i in indexes])]

ref_seqs = []
alt_seqs = []
five_p_seqs = []
three_p_seqs = []
for i, j, k, l in zip(refs, alts, five_ps, three_ps):
    flip = False
    if i[0] != '-' and i[0] in ['A', 'G']:
        flip = True
    if i[0] == '-' and j[0] in ['A', 'G']:
        flip = True
    if flip:
        ref_seqs.append(str(Seq(i).reverse_complement()))
        alt_seqs.append(str(Seq(j).reverse_complement()))
        five_p_seqs.append(str(Seq(l).reverse_complement()))
        three_p_seqs.append(str(Seq(k).reverse_complement()))
    else:
        ref_seqs.append(i)
        alt_seqs.append(j)
        five_p_seqs.append(k)
        three_p_seqs.append(l)


sbs_mask = [(len(i) == 1 and len(j) == 1 and len(re.findall('A|T|C|G', i)) == 1 and len(re.findall('A|T|C|G', j)) == 1) for i, j in zip(refs, alts)]
del_mask = [j == '-' for i, j in zip(refs, alts)]
ins_mask = [i == '-' for i, j in zip(refs, alts)]


##instance features
mask = samples.iloc[idx_test]['type'] == cancer
prediction = prediction_probabilities[mask][:, cancer_to_code[cancer]]
instances = np.concatenate(mil.hidden_model.predict(ds_test).numpy()[mask], axis=0)
instance_attention = np.concatenate(attention[mask], axis=0)[:, cancer_to_code[cancer]]
instance_attention = instance_attention - min(instance_attention)
matrix = instances.T
mask = np.sum(matrix, axis=-1) == 0
matrix = matrix[~mask]

kmeans = KMeans(n_clusters=8, random_state=0).fit(matrix.T)
cluster_similarity = [1 - spatial.distance.cosine(kmeans.cluster_centers_[0], cluster) for cluster in kmeans.cluster_centers_]
cluster_order = np.arange(8)[list(np.argsort(cluster_similarity))[::-1]]
label_dict = {i: j for i, j in zip(cluster_order, np.arange(8))}
labels = np.array([label_dict[i] for i in kmeans.labels_])
subsample = np.random.choice(np.arange(len(instances)), size=min(2000, len(instances)), replace=False)

submatrix = matrix.T[subsample][np.argsort(labels[subsample])].T

Z = linkage(submatrix, 'ward')
dn = dendrogram(Z, leaf_rotation=90, leaf_font_size=8, color_threshold=1)
submatrix = submatrix[list(dn.values())[3]]


label = kmeans.labels_[subsample][np.argsort(labels[subsample])]
var_type = np.array(sbs_mask) * 1 + np.array(del_mask) * 2 + np.array(ins_mask) * 3
var_type = var_type[subsample][np.argsort(labels[subsample])]

##
vmax = np.percentile(submatrix / submatrix.sum(axis=-1, keepdims=True), 99)
myblue = make_colormap({0: '#ffffff', vmax: '#4169E1'})

fig = plt.figure()
fig.subplots_adjust(left=.05,
                    bottom=.06,
                    right=.99,
                    top=.94,
                    hspace=.05)
gs = fig.add_gridspec(2, 1, height_ratios=[15, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

latent_matrix = ax1.imshow(submatrix / submatrix.sum(axis=-1, keepdims=True),
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

vmin = min(instance_attention[subsample])
vmax = np.percentile(instance_attention[subsample], 98)
myblue = make_colormap({vmin: '#ffffff', vmax * .6: '#e9eefc', vmax: '#4169E1'})
attention_matrix = ax2.imshow(instance_attention[subsample][np.argsort(labels[subsample])][np.newaxis, :],
                           cmap=myblue,
                           vmin=vmin,
                           vmax=vmax,
                            aspect='auto',
                          interpolation='nearest')

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_xlabel('Instances', fontsize=16)
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Features', fontsize=16)
ax2.set_xlabel('Attention', fontsize=16)




weighted_sbs_background_ref_matrix = pd.DataFrame(data={'C': np.repeat(0, 1), 'T': np.repeat(0, 1)})
weighted_sbs_background_alt_matrix = pd.DataFrame(data={'A': np.repeat(0, 1), 'C': np.repeat(0, 1), 'G': np.repeat(0, 1), 'T': np.repeat(0, 1)})
weighted_sbs_background_five_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)})
weighted_sbs_background_three_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)})

weighted_del_background_ref_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)})
weighted_del_background_five_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)})
weighted_del_background_three_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)})

weighted_ins_background_alt_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)})
weighted_ins_background_five_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)})
weighted_ins_background_three_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)})

for cancer in cancer_to_code:
    print(cancer)
    cancer_ref_seqs = []
    cancer_alt_seqs = []
    cancer_five_p_seqs = []
    cancer_three_p_seqs = []
    cancer_indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer].index]
    cancer_refs = tcga_maf['Reference_Allele'].values[np.concatenate([i[0] for i in cancer_indexes])]
    cancer_alts = tcga_maf['Tumor_Seq_Allele2'].values[np.concatenate([i[0] for i in cancer_indexes])]
    cancer_five_ps = tcga_maf['five_p'].values[np.concatenate([i[0] for i in cancer_indexes])]
    cancer_three_ps = tcga_maf['three_p'].values[np.concatenate([i[0] for i in cancer_indexes])]
    for i, j, k, l in zip(cancer_refs, cancer_alts, cancer_five_ps, cancer_three_ps):
        flip = False
        if i[0] != '-' and i[0] in ['A', 'G']:
            flip = True
        if i[0] == '-' and j[0] in ['A', 'G']:
            flip = True
        if flip:
            cancer_ref_seqs.append(str(Seq(i).reverse_complement()))
            cancer_alt_seqs.append(str(Seq(j).reverse_complement()))
            cancer_five_p_seqs.append(str(Seq(l).reverse_complement()))
            cancer_three_p_seqs.append(str(Seq(k).reverse_complement()))
        else:
            cancer_ref_seqs.append(i)
            cancer_alt_seqs.append(j)
            cancer_five_p_seqs.append(k)
            cancer_three_p_seqs.append(l)

    sbs_cancer_mask = [(len(i) == 1 and len(j) == 1 and len(re.findall('A|T|C|G', i)) == 1 and len(re.findall('A|T|C|G', j)) == 1) for i, j in zip(cancer_ref_seqs, cancer_alt_seqs)]
    del_cancer_mask = [j == '-' for i, j in zip(cancer_ref_seqs, cancer_alt_seqs)]
    ins_cancer_mask = [i == '-' for i, j in zip(cancer_ref_seqs, cancer_alt_seqs)]

    cancer_sbs_background_ref_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_ref_seqs, sbs_cancer_mask) if j])
    cancer_sbs_background_ref_matrix = lm.transform_matrix(cancer_sbs_background_ref_matrix, from_type='counts', to_type='probability')
    weighted_sbs_background_ref_matrix = weighted_sbs_background_ref_matrix + (cancer_sbs_background_ref_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_del_background_ref_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([(i + '-----')[:6] for i, j in zip(cancer_ref_seqs, del_cancer_mask) if j])
    cancer_del_background_ref_matrix = lm.transform_matrix(cancer_del_background_ref_matrix.fillna(0), from_type='counts', to_type='probability')
    weighted_del_background_ref_matrix = weighted_del_background_ref_matrix + (cancer_del_background_ref_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_sbs_background_alt_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_alt_seqs, sbs_cancer_mask) if j])
    cancer_sbs_background_alt_matrix = lm.transform_matrix(cancer_sbs_background_alt_matrix, from_type='counts', to_type='probability')
    weighted_sbs_background_alt_matrix = weighted_sbs_background_alt_matrix + (cancer_sbs_background_alt_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_ins_background_alt_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([(i + '-----')[:6] for i, j in zip(cancer_alt_seqs, ins_cancer_mask) if j])
    cancer_ins_background_alt_matrix = lm.transform_matrix(cancer_ins_background_alt_matrix.fillna(0), from_type='counts', to_type='probability')
    weighted_ins_background_alt_matrix = weighted_ins_background_alt_matrix + (cancer_ins_background_alt_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_sbs_background_five_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_five_p_seqs, sbs_cancer_mask) if j])
    cancer_sbs_background_five_p_matrix = lm.transform_matrix(cancer_sbs_background_five_p_matrix, from_type='counts', to_type='probability')
    weighted_sbs_background_five_p_matrix = weighted_sbs_background_five_p_matrix + (cancer_sbs_background_five_p_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_del_background_five_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_five_p_seqs, del_cancer_mask) if j])
    cancer_del_background_five_p_matrix = lm.transform_matrix(cancer_del_background_five_p_matrix, from_type='counts', to_type='probability')
    weighted_del_background_five_p_matrix = weighted_del_background_five_p_matrix + (cancer_del_background_five_p_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_ins_background_five_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_five_p_seqs, ins_cancer_mask) if j])
    cancer_ins_background_five_p_matrix = lm.transform_matrix(cancer_ins_background_five_p_matrix, from_type='counts', to_type='probability')
    weighted_ins_background_five_p_matrix = weighted_ins_background_five_p_matrix + (cancer_ins_background_five_p_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_sbs_background_three_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_three_p_seqs, sbs_cancer_mask) if j])
    cancer_sbs_background_three_p_matrix = lm.transform_matrix(cancer_sbs_background_three_p_matrix, from_type='counts', to_type='probability')
    weighted_sbs_background_three_p_matrix = weighted_sbs_background_three_p_matrix + (cancer_sbs_background_three_p_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_del_background_three_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_three_p_seqs, del_cancer_mask) if j])
    cancer_del_background_three_p_matrix = lm.transform_matrix(cancer_del_background_three_p_matrix, from_type='counts', to_type='probability')
    weighted_del_background_three_p_matrix = weighted_del_background_three_p_matrix + (cancer_del_background_three_p_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))

    cancer_ins_background_three_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_three_p_seqs, ins_cancer_mask) if j])
    cancer_ins_background_three_p_matrix = lm.transform_matrix(cancer_ins_background_three_p_matrix, from_type='counts', to_type='probability')
    weighted_ins_background_three_p_matrix = weighted_ins_background_three_p_matrix + (cancer_ins_background_three_p_matrix * len(samples.iloc[idx_test].loc[samples.iloc[idx_test]['type'] == cancer]))


weighted_sbs_background_ref_matrix = weighted_sbs_background_ref_matrix / len(idx_test)
weighted_sbs_background_alt_matrix = weighted_sbs_background_alt_matrix / len(idx_test)
weighted_sbs_background_five_p_matrix = weighted_sbs_background_five_p_matrix / len(idx_test)
weighted_sbs_background_three_p_matrix = weighted_sbs_background_three_p_matrix / len(idx_test)

weighted_del_background_ref_matrix = weighted_del_background_ref_matrix / len(idx_test)
weighted_del_background_five_p_matrix = weighted_del_background_five_p_matrix / len(idx_test)
weighted_del_background_three_p_matrix = weighted_del_background_three_p_matrix / len(idx_test)

weighted_ins_background_alt_matrix = weighted_ins_background_alt_matrix / len(idx_test)
weighted_ins_background_five_p_matrix = weighted_ins_background_five_p_matrix / len(idx_test)
weighted_ins_background_three_p_matrix = weighted_ins_background_three_p_matrix / len(idx_test)


##cluster logos
cluster = 3
sbs_ref_matrix = pd.DataFrame(data={'C': np.repeat(0, 1), 'T': np.repeat(0, 1)}) + lm.alignment_to_matrix([i for i, j, k in zip(ref_seqs, sbs_mask, kmeans.labels_) if (j and k == cluster)])
sbs_ref_matrix = lm.transform_matrix(sbs_ref_matrix.fillna(0), from_type='counts', to_type='probability')
sbs_alt_matrix = pd.DataFrame(data={'A': np.repeat(0, 1), 'C': np.repeat(0, 1), 'G': np.repeat(0, 1), 'T': np.repeat(0, 1)}) + lm.alignment_to_matrix([i for i, j, k in zip(alt_seqs, sbs_mask, kmeans.labels_) if (j and k == cluster)])
sbs_alt_matrix = lm.transform_matrix(sbs_alt_matrix.fillna(0), from_type='counts', to_type='probability')
sbs_five_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([i for i, j, k in zip(five_p_seqs, sbs_mask, kmeans.labels_) if (j and k == cluster)])
sbs_five_p_matrix = lm.transform_matrix(sbs_five_p_matrix.fillna(0), from_type='counts', to_type='probability')
sbs_three_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([i for i, j, k in zip(three_p_seqs, sbs_mask, kmeans.labels_) if (j and k == cluster)])
sbs_three_p_matrix = lm.transform_matrix(sbs_three_p_matrix.fillna(0), from_type='counts', to_type='probability')

lm.Logo(lm.transform_matrix(sbs_five_p_matrix, from_type='probability', to_type='information', background=weighted_sbs_background_five_p_matrix))
lm.Logo(lm.transform_matrix(sbs_ref_matrix, from_type='probability', to_type='information', background=weighted_sbs_background_ref_matrix))
lm.Logo(lm.transform_matrix(sbs_alt_matrix, from_type='probability', to_type='information', background=weighted_sbs_background_alt_matrix))
lm.Logo(lm.transform_matrix(sbs_three_p_matrix, from_type='probability', to_type='information', background=weighted_sbs_background_three_p_matrix))

fig = plt.figure()
fig.subplots_adjust(left=.07,
                    right=1,
                    bottom=.15,
                    top=.5
                    )
gs = fig.add_gridspec(1, 4, width_ratios=[6, 1, 1, 6])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
lm.Logo(lm.transform_matrix(sbs_five_p_matrix, from_type='probability', to_type='information', background=weighted_sbs_background_five_p_matrix), ax=ax1)
lm.Logo(lm.transform_matrix(sbs_ref_matrix, from_type='probability', to_type='information', background=weighted_sbs_background_ref_matrix), ax=ax2)
lm.Logo(lm.transform_matrix(sbs_alt_matrix, from_type='probability', to_type='information', background=weighted_sbs_background_alt_matrix), ax=ax3)
lm.Logo(lm.transform_matrix(sbs_three_p_matrix, from_type='probability', to_type='information', background=weighted_sbs_background_three_p_matrix), ax=ax4)
for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(0, 2.2)
    ax.tick_params(axis='x', length=0, width=0, labelsize=8)
ax1.tick_params(axis='y', length=0, width=0, labelsize=8)
ax1.set_yticks([0, .5, 1, 1.5, 2])
for ax in [ax2, ax3, ax4]:
    ax.set_yticks([])
ax1.set_xticks(list(range(6)))
ax1.set_xticklabels([-6, -5, -4, -3, -2, -1])
ax4.set_xticks(list(range(6)))
ax4.set_xticklabels(['+1', '+2', '+3', '+4', '+5', '+6'])
ax2.set_xticks([])
ax3.set_xticks([])
ax1.set_xlabel("Five prime", fontsize=12)
ax2.set_xlabel("Ref", fontsize=12)
ax3.set_xlabel("Alt", fontsize=12)
ax4.set_xlabel("Three prime", fontsize=12)
ax1.set_ylabel("Bits", fontsize=12)
ax2.text(.6, -.2, '>', fontsize=12)



# [i for i, j, k in zip(tcga_maf['Hugo_Symbol'].values[np.concatenate([i[0] for i in indexes])], sbs_mask, kmeans.labels_) if (j and k == cluster)]