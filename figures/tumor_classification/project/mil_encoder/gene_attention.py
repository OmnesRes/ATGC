import numpy as np
from model.Sample_MIL import RaggedModels, InstanceModels
import tensorflow as tf
from model import DatasetsUtils
import pickle
import pathlib

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
mil = RaggedModels.MIL(instance_encoders=[gene_encoder.model], sample_encoders=[], heads=y_label.shape[-1], output_types=['other'], mil_hidden=[], attention_layers=[], instance_dropout=0, regularization=0, input_dropout=.5)
mil.model.set_weights(weights[0])

idx_test = test_idx[0]
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

cancer_to_code = {cancer:index for index, cancer in enumerate(A.cat.categories)}

import pylab as plt
for cancer in cancer_to_code:
    fig = plt.figure()
    cancer_attention = np.concatenate([i[:, cancer_to_code[cancer]] for i in attention])
    plt.hist(cancer_attention, bins=100)
    plt.title(cancer)
    plt.show()


cancer_indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].index]
cancer_genes = tcga_maf['Hugo_Symbol'].values[np.concatenate([i[0] for i in cancer_indexes])]
cancer_gene_attention = {}
for cancer in cancer_to_code:
    cancer_attention = np.concatenate([i[:, cancer_to_code[cancer]] for i in attention])
    cutoff = np.percentile(cancer_attention, 99)
    cancer_gene_attention[cancer] = cancer_genes[cancer_attention > cutoff]

all_genes = np.unique(np.concatenate([np.unique(cancer_gene_attention[cancer]) for cancer in cancer_gene_attention]))
# for cancer in cancer_to_code:
#     print(cancer, np.unique(cancer_gene_attention[cancer], return_counts=True))