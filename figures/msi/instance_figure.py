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
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-2], True)
tf.config.experimental.set_visible_devices(physical_devices[-2], 'GPU')

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))
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

idx_test = test_idx[fold]
test_indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].index]
refs = tcga_maf['Reference_Allele'].values[np.concatenate([i[0] for i in test_indexes])]
alts = tcga_maf['Tumor_Seq_Allele2'].values[np.concatenate([i[0] for i in test_indexes])]
five_ps = tcga_maf['five_p'].values[np.concatenate([i[0] for i in test_indexes])]
three_ps = tcga_maf['three_p'].values[np.concatenate([i[0] for i in test_indexes])]


sbs_mask = [(len(i) == 1 and len(j) == 1 and len(re.findall('A|T|C|G', i)) == 1 and len(re.findall('A|T|C|G', j)) == 1) for i, j in zip(refs, alts)]
del_mask = [j == '-' for i, j in zip(refs, alts)]
ins_mask = [i == '-' for i, j in zip(refs, alts)]

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
        ref_seqs.append(str(Seq(i).reverse_complement()).replace('N', '-'))
        alt_seqs.append(str(Seq(j).reverse_complement()).replace('N', '-'))
        five_p_seqs.append(str(Seq(l).reverse_complement()).replace('N', '-'))
        three_p_seqs.append(str(Seq(k).reverse_complement()).replace('N', '-'))
    else:
        ref_seqs.append(i.replace('N', '-'))
        alt_seqs.append(j.replace('N', '-'))
        five_p_seqs.append(k.replace('N', '-'))
        three_p_seqs.append(l.replace('N', '-'))

##instance features
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
instances = np.concatenate(mil.hidden_model.predict(ds_test).numpy(), axis=0)
prediction_probabilities = tf.nn.sigmoid(mil.model.predict(ds_test))
instance_attention = np.concatenate(attention, axis=0)[:, 0]
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
mask = np.sum(submatrix, axis=-1) == 0
submatrix = submatrix[~mask]

Z = linkage(submatrix, 'ward')
dn = dendrogram(Z, leaf_rotation=90, leaf_font_size=8, color_threshold=1)
submatrix = submatrix[list(dn.values())[3]]

label = kmeans.labels_[subsample][np.argsort(labels[subsample])]

##
vmax = np.percentile(submatrix / submatrix.sum(axis=-1, keepdims=True), 99)
myblue = make_colormap({0: '#ffffff', vmax: '#4169E1'})

fig = plt.figure()
fig.subplots_adjust(left=.05,
                    bottom=.055,
                    right=.99,
                    top=.94,
                    hspace=.04)
gs = fig.add_gridspec(3, 1, height_ratios=[25, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

latent_matrix = ax1.imshow(submatrix / submatrix.sum(axis=-1, keepdims=True),
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

myqual = make_colormap({0: '#d53e4f', 7: '#f46d43', 5: '#fdae61', 3: '#fee08b', 1: '#e6f598', 4: '#abdda4', 6: '#66c2a5', 2: '#3288bd'})

ax2.imshow(label[np.newaxis, :],
                           cmap=myqual,
                          aspect='auto',
                          interpolation='nearest')

vmin = min(instance_attention[subsample])
vmax = np.percentile(instance_attention[subsample], 98)

myblue = make_colormap({vmin: '#ffffff', vmax * .6: '#e9eefc', vmax: '#4169E1'})
attention_matrix = ax3.imshow(instance_attention[subsample][np.argsort(labels[subsample])][np.newaxis, :],
                           cmap=myblue,
                           vmin=vmin,
                           vmax=vmax,
                            aspect='auto',
                          interpolation='nearest')

for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
ax1.set_xlabel('Instances', fontsize=16)
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Features', fontsize=16)
ax3.set_xlabel('Attention', fontsize=16)

plt.savefig(cwd / 'figures' / 'msi' / 'instance_figure.png', dpi=600)

##5, 2
# [sum(D['repeat'][np.concatenate([i[0] for i in test_indexes])][kmeans.labels_ == i]) / sum(kmeans.labels_ == i) for i in range(8)]
test = tcga_maf.iloc[np.concatenate([i[0] for i in test_indexes])]
test = tcga_maf.iloc[np.concatenate([i[0] for i in test_indexes])[kmeans.labels_ == 2]]
# test[['Variant_Classification', 'repeat']].value_counts()

# [i[-5:] for i,j in zip(five_p_seqs, kmeans.labels_) if j==0][:100]

del_background_ref_matrix = lm.alignment_to_matrix([(i + '-------------------')[:20] for i, j in zip(ref_seqs, del_mask) if j])
del_background_ref_matrix = lm.transform_matrix(del_background_ref_matrix, from_type='counts', to_type='probability')

del_background_five_p_matrix = lm.alignment_to_matrix([i for i, j in zip(five_p_seqs, del_mask) if j])
del_background_five_p_matrix = lm.transform_matrix(del_background_five_p_matrix, from_type='counts', to_type='probability')

del_background_three_p_matrix = lm.alignment_to_matrix([i for i, j in zip(three_p_seqs, del_mask) if j])
del_background_three_p_matrix = lm.transform_matrix(del_background_three_p_matrix, from_type='counts', to_type='probability')

##cluster logos
cluster = 2
del_ref_matrix = pd.DataFrame(data={'A': np.repeat(0, 20), 'C': np.repeat(0, 20), 'G': np.repeat(0, 20), 'T': np.repeat(0, 20)}) + lm.alignment_to_matrix([(i + '-------------------')[:20] for i, j, k in zip(ref_seqs, del_mask, kmeans.labels_) if (j and k == cluster)])
del_ref_matrix = lm.transform_matrix(del_ref_matrix.fillna(0), from_type='counts', to_type='probability')
del_five_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 20), 'C': np.repeat(0, 20), 'G': np.repeat(0, 20), 'T': np.repeat(0, 20)}) + lm.alignment_to_matrix([i for i, j, k in zip(five_p_seqs, del_mask, kmeans.labels_) if (j and k == cluster)])
del_five_p_matrix = lm.transform_matrix(del_five_p_matrix.fillna(0), from_type='counts', to_type='probability')
del_three_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 20), 'C': np.repeat(0, 20), 'G': np.repeat(0, 20), 'T': np.repeat(0, 20)}) + lm.alignment_to_matrix([i for i, j, k in zip(three_p_seqs, del_mask, kmeans.labels_) if (j and k == cluster)])
del_three_p_matrix = lm.transform_matrix(del_three_p_matrix.fillna(0), from_type='counts', to_type='probability')

lm.Logo(lm.transform_matrix(del_five_p_matrix, from_type='probability', to_type='information', background=del_background_five_p_matrix), color_scheme='classic')
lm.Logo(lm.transform_matrix(del_ref_matrix, from_type='probability', to_type='information', background=del_background_ref_matrix), color_scheme='classic')
lm.Logo(lm.transform_matrix(del_three_p_matrix, from_type='probability', to_type='information', background=del_background_three_p_matrix), color_scheme='classic')

lm.Logo(del_five_p_matrix, color_scheme='classic')
lm.Logo(del_ref_matrix, color_scheme='classic')
lm.Logo(del_three_p_matrix, color_scheme='classic')


fig = plt.figure()
fig.subplots_adjust(left=.08,
                    right=.99,
                    bottom=.15,
                    top=.5,
                    wspace=.023
                    )
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
lm.Logo(lm.transform_matrix(del_five_p_matrix, from_type='probability', to_type='information', background=del_background_five_p_matrix), ax=ax1, color_scheme='classic')
lm.Logo(lm.transform_matrix(del_ref_matrix, from_type='probability', to_type='information', background=del_background_ref_matrix), ax=ax2, color_scheme='classic')
lm.Logo(lm.transform_matrix(del_three_p_matrix, from_type='probability', to_type='information', background=del_background_three_p_matrix), ax=ax3, color_scheme='classic')
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(0, .75)
    ax.tick_params(axis='x', length=0, width=0, labelsize=8)
ax1.tick_params(axis='y', length=0, width=0, labelsize=8)
ax1.set_yticks([0, .25, .5, .75])
for ax in [ax2, ax3]:
    ax.set_yticks([])
ax1.set_xticks(list(range(0, 20, 5)))
ax1.set_xticklabels(list(range(-20, 0, 5)))
ax2.set_xticks([])
ax3.set_xticks(list(range(4, 24, 5)))
ax3.set_xticklabels(['+' + str(i + 1) for i in range(4, 20, 5)])
ax1.set_xlabel("Five prime", fontsize=12)
ax2.set_xlabel("Ref", fontsize=12)
ax3.set_xlabel("Three prime", fontsize=12)
ax1.set_ylabel("Bits", fontsize=12)
plt.savefig(cwd / 'figures' / 'msi' / 'logo_del_cluster.pdf')

