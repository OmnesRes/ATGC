import numpy as np
from Bio.Seq import Seq
from model.Sample_MIL import RaggedModels, InstanceModels
import tensorflow as tf
from model import DatasetsUtils
import pickle
import pathlib
import logomaker as lm

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
cancer_to_code = {cancer:index for index, cancer in enumerate(A.cat.categories)}

#
# import pylab as plt
# for cancer in range(24):
#     fig = plt.figure()
#     cancer_attention = np.concatenate([i[:, cancer] for i in attention])
#     plt.hist(cancer_attention, bins=100)
#     plt.show()


cancer_indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].index]
refs = tcga_maf['Reference_Allele'].values[np.concatenate([i[0] for i in cancer_indexes])]
alts = tcga_maf['Tumor_Seq_Allele2'].values[np.concatenate([i[0] for i in cancer_indexes])]
five_ps = tcga_maf['five_p'].values[np.concatenate([i[0] for i in cancer_indexes])]
three_ps = tcga_maf['three_p'].values[np.concatenate([i[0] for i in cancer_indexes])]


cancer_attention_refs = {}
cancer_attention_alts = {}
cancer_attention_five_ps = {}
cancer_attention_three_ps = {}

for cancer in cancer_to_code:
    cancer_attention = np.concatenate([i[:, cancer_to_code[cancer]] for i in attention])
    cutoff = np.percentile(cancer_attention, 95)
    cancer_attention_refs[cancer] = refs[cancer_attention > cutoff]
    cancer_attention_alts[cancer] = alts[cancer_attention > cutoff]
    cancer_attention_five_ps[cancer] = five_ps[cancer_attention > cutoff]
    cancer_attention_three_ps[cancer] = three_ps[cancer_attention > cutoff]


five_p_seqs = []
three_p_seqs = []
five_p_seqs = []
five_p_seqs = []

counts={}
for i, j, k, l in zip(refs, alts, five_ps, three_ps):
    flip = False
    if i[0] != '-' and i[0] in ['A', 'G']:
        flip = True
    if i[0] == '-' and j[0] in ['A', 'G']:
        flip = True
    if flip:
        counts[str(Seq(l).reverse_complement())[-1] + '_' + str(Seq(i).reverse_complement()) + '_' + str(Seq(j).reverse_complement()) + '_' + str(Seq(k).reverse_complement())[0]] = \
            counts.get(str(Seq(l).reverse_complement())[-1]  + '_' + str(Seq(i).reverse_complement()) + '_' + str(Seq(j).reverse_complement()) + '_' + str(Seq(k).reverse_complement())[0], 0) + 1
    else:
        counts[k[-1]  + '_' + i + '_' + j + '_' + l[0]] = counts.get(k[-1]  + '_' + i + '_' + j + '_' + l[0], 0) + 1

seqs = np.array(list(counts.keys()))[np.argsort(list(counts.values()))]
print(seqs[-12:])
print(np.array(list(counts.values()))[np.argsort(list(counts.values()))][-12:] / np.sum(list(counts.values())) * 100)


for cancer in cancer_to_code:
    print(cancer)
    counts = {}
    for i, j, k, l in zip(cancer_attention_refs[cancer], cancer_attention_alts[cancer], cancer_attention_five_ps[cancer], cancer_attention_three_ps[cancer]):
        flip = False
        if i[0] != '-' and i[0] in ['A', 'G']:
            flip = True
        if i[0] == '-' and j[0] in ['A', 'G']:
            flip = True
        if flip:
            counts[str(Seq(l).reverse_complement())[-1] + '_' + str(Seq(i).reverse_complement()) + '_' + str(Seq(j).reverse_complement()) + '_' + str(Seq(k).reverse_complement())[0]] = \
                counts.get(str(Seq(l).reverse_complement())[-1] + '_' + str(Seq(i).reverse_complement()) + '_' + str(Seq(j).reverse_complement()) + '_' + str(Seq(k).reverse_complement())[0], 0) + 1
        else:
            counts[k[-1] + '_' + i + '_' + j + '_' + l[0]] = counts.get(k[-1] + '_' + i + '_' + j + '_' + l[0], 0) + 1
    seqs = np.array(list(counts.keys()))[np.argsort(list(counts.values()))]
    print(seqs[-12:])
    print(np.array(list(counts.values()))[np.argsort(list(counts.values()))][-12:] / np.sum(list(counts.values())) * 100)






