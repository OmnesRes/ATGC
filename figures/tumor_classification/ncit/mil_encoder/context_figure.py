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
import matplotlib as mpl

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

A = samples['NCIt_label'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot

y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)
y_weights_loader = DatasetsUtils.Map.FromNumpy(y_weights, tf.float32)

test_idx, weights = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'mil_encoder' / 'results' / 'context_weights.pkl', 'rb'))
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
ds_test = ds_test.batch(500, drop_remainder=False)
attention = mil.attention_model.predict(ds_test).numpy()
cancer_to_code = {cancer: index for index, cancer in enumerate(A.cat.categories)}

test_indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].index]
refs = tcga_maf['Reference_Allele'].values[np.concatenate([i[0] for i in test_indexes])]
alts = tcga_maf['Tumor_Seq_Allele2'].values[np.concatenate([i[0] for i in test_indexes])]
five_ps = tcga_maf['five_p'].values[np.concatenate([i[0] for i in test_indexes])]
three_ps = tcga_maf['three_p'].values[np.concatenate([i[0] for i in test_indexes])]
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

sbs_mask = [(len(i) == 1 and len(j) == 1 and len(re.findall('A|T|C|G', i)) == 1 and len(re.findall('A|T|C|G', j)) == 1) for i, j in zip(ref_seqs, alt_seqs)]
del_mask = [j == '-' for i, j in zip(ref_seqs, alt_seqs)]
ins_mask = [i == '-' for i, j in zip(ref_seqs, alt_seqs)]

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

sbs_ref_matrices = {}
sbs_alt_matrices = {}
sbs_five_p_matrices = {}
sbs_three_p_matrices = {}

del_ref_matrices = {}
del_five_p_matrices = {}
del_three_p_matrices = {}

ins_alt_matrices = {}
ins_five_p_matrices = {}
ins_three_p_matrices = {}

for cancer in cancer_to_code:
    print(cancer)
    cancer_ref_seqs = []
    cancer_alt_seqs = []
    cancer_five_p_seqs = []
    cancer_three_p_seqs = []
    cancer_indexes = [np.where(D['sample_idx'] == idx) for idx in samples.iloc[idx_test].loc[samples.iloc[idx_test]['NCIt_label'] == cancer].index]
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

    cancer_attention = np.concatenate([i[:, cancer_to_code[cancer]] for i in attention])

    sbs_cutoff = np.percentile(cancer_attention[sbs_mask], 95)
    del_cutoff = np.percentile(cancer_attention[del_mask], 95)
    ins_cutoff = np.percentile(cancer_attention[ins_mask], 95)

    sbs_cancer_ref_matrix = pd.DataFrame(data={'C': np.repeat(0, 1), 'T': np.repeat(0, 1)}) + lm.alignment_to_matrix([i for i, j, k in zip(ref_seqs, sbs_mask, cancer_attention) if (j and k > sbs_cutoff)])
    sbs_cancer_ref_matrix = lm.transform_matrix(sbs_cancer_ref_matrix.fillna(0), from_type='counts', to_type='probability')
    sbs_ref_matrices[cancer] = sbs_cancer_ref_matrix

    del_cancer_ref_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([(i + '-----')[:6] for i, j, k in zip(ref_seqs, del_mask, cancer_attention) if (j and k > del_cutoff)])
    del_cancer_ref_matrix = lm.transform_matrix(del_cancer_ref_matrix.fillna(0), from_type='counts', to_type='probability')
    del_ref_matrices[cancer] = del_cancer_ref_matrix

    cancer_sbs_background_ref_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_ref_seqs, sbs_cancer_mask) if j])
    cancer_sbs_background_ref_matrix = lm.transform_matrix(cancer_sbs_background_ref_matrix, from_type='counts', to_type='probability')
    weighted_sbs_background_ref_matrix = weighted_sbs_background_ref_matrix + cancer_sbs_background_ref_matrix

    cancer_del_background_ref_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([(i + '-----')[:6] for i, j in zip(cancer_ref_seqs, del_cancer_mask) if j])
    cancer_del_background_ref_matrix = lm.transform_matrix(cancer_del_background_ref_matrix.fillna(0), from_type='counts', to_type='probability')
    weighted_del_background_ref_matrix = weighted_del_background_ref_matrix + cancer_del_background_ref_matrix

    sbs_cancer_alt_matrix = pd.DataFrame(data={'A': np.repeat(0, 1), 'C': np.repeat(0, 1), 'G': np.repeat(0, 1), 'T': np.repeat(0, 1)}) + lm.alignment_to_matrix([i for i, j, k in zip(alt_seqs, sbs_mask, cancer_attention) if (j and k > sbs_cutoff)])
    sbs_cancer_alt_matrix = lm.transform_matrix(sbs_cancer_alt_matrix.fillna(0), from_type='counts', to_type='probability')
    sbs_alt_matrices[cancer] = sbs_cancer_alt_matrix

    ins_cancer_alt_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([(i + '-----')[:6] for i, j, k in zip(alt_seqs, ins_mask, cancer_attention) if (j and k > ins_cutoff)])
    ins_cancer_alt_matrix = lm.transform_matrix(ins_cancer_alt_matrix.fillna(0), from_type='counts', to_type='probability')
    ins_alt_matrices[cancer] = ins_cancer_alt_matrix

    cancer_sbs_background_alt_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_alt_seqs, sbs_cancer_mask) if j])
    cancer_sbs_background_alt_matrix = lm.transform_matrix(cancer_sbs_background_alt_matrix, from_type='counts', to_type='probability')
    weighted_sbs_background_alt_matrix = weighted_sbs_background_alt_matrix + cancer_sbs_background_alt_matrix

    cancer_ins_background_alt_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([(i + '-----')[:6] for i, j in zip(cancer_alt_seqs, ins_cancer_mask) if j])
    cancer_ins_background_alt_matrix = lm.transform_matrix(cancer_ins_background_alt_matrix.fillna(0), from_type='counts', to_type='probability')
    weighted_ins_background_alt_matrix = weighted_ins_background_alt_matrix + cancer_ins_background_alt_matrix

    sbs_cancer_five_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([i for i, j, k in zip(five_p_seqs, sbs_mask, cancer_attention) if (j and k > sbs_cutoff)])
    sbs_cancer_five_p_matrix = lm.transform_matrix(sbs_cancer_five_p_matrix.fillna(0), from_type='counts', to_type='probability')
    sbs_five_p_matrices[cancer] = sbs_cancer_five_p_matrix

    del_cancer_five_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([i for i, j, k in zip(five_p_seqs, del_mask, cancer_attention) if (j and k > del_cutoff)])
    del_cancer_five_p_matrix = lm.transform_matrix(del_cancer_five_p_matrix.fillna(0), from_type='counts', to_type='probability')
    del_five_p_matrices[cancer] = del_cancer_five_p_matrix

    ins_cancer_five_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([i for i, j, k in zip(five_p_seqs, ins_mask, cancer_attention) if (j and k > ins_cutoff)])
    ins_cancer_five_p_matrix = lm.transform_matrix(ins_cancer_five_p_matrix.fillna(0), from_type='counts', to_type='probability')
    ins_five_p_matrices[cancer] = ins_cancer_five_p_matrix

    cancer_sbs_background_five_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_five_p_seqs, sbs_cancer_mask) if j])
    cancer_sbs_background_five_p_matrix = lm.transform_matrix(cancer_sbs_background_five_p_matrix, from_type='counts', to_type='probability')
    weighted_sbs_background_five_p_matrix = weighted_sbs_background_five_p_matrix + cancer_sbs_background_five_p_matrix

    cancer_del_background_five_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_five_p_seqs, del_cancer_mask) if j])
    cancer_del_background_five_p_matrix = lm.transform_matrix(cancer_del_background_five_p_matrix, from_type='counts', to_type='probability')
    weighted_del_background_five_p_matrix = weighted_del_background_five_p_matrix + cancer_del_background_five_p_matrix

    cancer_ins_background_five_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_five_p_seqs, ins_cancer_mask) if j])
    cancer_ins_background_five_p_matrix = lm.transform_matrix(cancer_ins_background_five_p_matrix, from_type='counts', to_type='probability')
    weighted_ins_background_five_p_matrix = weighted_ins_background_five_p_matrix + cancer_ins_background_five_p_matrix

    sbs_cancer_three_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([i for i, j, k in zip(three_p_seqs, sbs_mask, cancer_attention) if (j and k > sbs_cutoff)])
    sbs_cancer_three_p_matrix = lm.transform_matrix(sbs_cancer_three_p_matrix.fillna(0), from_type='counts', to_type='probability')
    sbs_three_p_matrices[cancer] = sbs_cancer_three_p_matrix

    del_cancer_three_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([i for i, j, k in zip(three_p_seqs, del_mask, cancer_attention) if (j and k > del_cutoff)])
    del_cancer_three_p_matrix = lm.transform_matrix(del_cancer_three_p_matrix.fillna(0), from_type='counts', to_type='probability')
    del_three_p_matrices[cancer] = del_cancer_three_p_matrix

    ins_cancer_three_p_matrix = pd.DataFrame(data={'A': np.repeat(0, 6), 'C': np.repeat(0, 6), 'G': np.repeat(0, 6), 'T': np.repeat(0, 6)}) + lm.alignment_to_matrix([i for i, j, k in zip(three_p_seqs, ins_mask, cancer_attention) if (j and k > ins_cutoff)])
    ins_cancer_three_p_matrix = lm.transform_matrix(ins_cancer_three_p_matrix.fillna(0), from_type='counts', to_type='probability')
    ins_three_p_matrices[cancer] = ins_cancer_three_p_matrix

    cancer_sbs_background_three_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_three_p_seqs, sbs_cancer_mask) if j])
    cancer_sbs_background_three_p_matrix = lm.transform_matrix(cancer_sbs_background_three_p_matrix, from_type='counts', to_type='probability')
    weighted_sbs_background_three_p_matrix = weighted_sbs_background_three_p_matrix + cancer_sbs_background_three_p_matrix

    cancer_del_background_three_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_three_p_seqs, del_cancer_mask) if j])
    cancer_del_background_three_p_matrix = lm.transform_matrix(cancer_del_background_three_p_matrix, from_type='counts', to_type='probability')
    weighted_del_background_three_p_matrix = weighted_del_background_three_p_matrix + cancer_del_background_three_p_matrix

    cancer_ins_background_three_p_matrix = lm.alignment_to_matrix([i for i, j in zip(cancer_three_p_seqs, ins_cancer_mask) if j])
    cancer_ins_background_three_p_matrix = lm.transform_matrix(cancer_ins_background_three_p_matrix, from_type='counts', to_type='probability')
    weighted_ins_background_three_p_matrix = weighted_ins_background_three_p_matrix + cancer_ins_background_three_p_matrix

weighted_sbs_background_ref_matrix = weighted_sbs_background_ref_matrix / len(cancer_to_code)
weighted_sbs_background_alt_matrix = weighted_sbs_background_alt_matrix / len(cancer_to_code)
weighted_sbs_background_five_p_matrix = weighted_sbs_background_five_p_matrix / len(cancer_to_code)
weighted_sbs_background_three_p_matrix = weighted_sbs_background_three_p_matrix / len(cancer_to_code)

weighted_del_background_ref_matrix = weighted_del_background_ref_matrix / len(cancer_to_code)
weighted_del_background_five_p_matrix = weighted_del_background_five_p_matrix / len(cancer_to_code)
weighted_del_background_three_p_matrix = weighted_del_background_three_p_matrix / len(cancer_to_code)

weighted_ins_background_alt_matrix = weighted_ins_background_alt_matrix / len(cancer_to_code)
weighted_ins_background_five_p_matrix = weighted_ins_background_five_p_matrix / len(cancer_to_code)
weighted_ins_background_three_p_matrix = weighted_ins_background_three_p_matrix / len(cancer_to_code)

five_p_sbs_information = []
three_p_sbs_information = []
ref_sbs_information = []
alt_sbs_information = []

five_p_del_information = []
three_p_del_information = []
ref_del_information = []

five_p_ins_information = []
three_p_ins_information = []
alt_ins_information = []

for cancer in cancer_to_code:
    five_p_sbs_information.append(np.sum(lm.transform_matrix(sbs_five_p_matrices[cancer], from_type='probability', to_type='information', background=weighted_sbs_background_five_p_matrix).values, axis=-1))
    ref_sbs_information.append(np.sum(lm.transform_matrix(sbs_ref_matrices[cancer], from_type='probability', to_type='information', background=weighted_sbs_background_ref_matrix).values, axis=-1))
    alt_sbs_information.append(np.sum(lm.transform_matrix(sbs_alt_matrices[cancer], from_type='probability', to_type='information', background=weighted_sbs_background_alt_matrix).values, axis=-1))
    three_p_sbs_information.append(np.sum(lm.transform_matrix(sbs_three_p_matrices[cancer], from_type='probability', to_type='information', background=weighted_sbs_background_three_p_matrix).values, axis=-1))

    five_p_del_information.append(np.sum(lm.transform_matrix(del_five_p_matrices[cancer], from_type='probability', to_type='information', background=weighted_del_background_five_p_matrix).values, axis=-1))
    ref_del_information.append(np.sum(lm.transform_matrix(del_ref_matrices[cancer], from_type='probability', to_type='information', background=weighted_del_background_ref_matrix).values, axis=-1))
    three_p_del_information.append(np.sum(lm.transform_matrix(del_three_p_matrices[cancer], from_type='probability', to_type='information', background=weighted_del_background_three_p_matrix).values, axis=-1))

    five_p_ins_information.append(np.sum(lm.transform_matrix(ins_five_p_matrices[cancer], from_type='probability', to_type='information', background=weighted_ins_background_five_p_matrix).values, axis=-1))
    alt_ins_information.append(np.sum(lm.transform_matrix(ins_alt_matrices[cancer], from_type='probability', to_type='information', background=weighted_ins_background_alt_matrix).values, axis=-1))
    three_p_ins_information.append(np.sum(lm.transform_matrix(ins_three_p_matrices[cancer], from_type='probability', to_type='information', background=weighted_ins_background_three_p_matrix).values, axis=-1))



def make_colormap(colors):
    z = np.sort(list(colors.keys()))
    anchors = (z - min(z)) / (max(z) - min(z))
    CC = mpl.colors.ColorConverter()
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
    mymap = mpl.colors.LinearSegmentedColormap('mymap', cmap_dict)
    return mymap


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

##sbs
vmax = np.max([np.max(np.max(five_p_sbs_information, axis=-1)), np.max(np.max(three_p_sbs_information, axis=-1)), np.max(np.max(ref_sbs_information, axis=-1)), np.max(np.max(alt_sbs_information, axis=-1))])
myblue = make_colormap({0: '#ffffff', vmax * .5: '#91a8ee', vmax: '#4169E1'})

fig = plt.figure()
fig.subplots_adjust(left=.056,
                    bottom=.06,
                    right=.95,
                    top=.94,
                    wspace=.1)
gs = fig.add_gridspec(1, 5, width_ratios=[6, 1, 1, 6, .5])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])

ax1.imshow(np.array(five_p_sbs_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

ax2.imshow(np.array(ref_sbs_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

ax3.imshow(np.array(alt_sbs_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

ax4.imshow(np.array(three_p_sbs_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

for ax in [ax1, ax2, ax3, ax4]:
    ax.xaxis.set_label_position('top')
ax1.set_xticks(list(range(6)))
ax1.set_xticklabels([-6, -5, -4, -3, -2, -1])
ax4.set_xticks(list(range(6)))
ax4.set_xticklabels(['+1', '+2', '+3', '+4', '+5', '+6'])
ax1.set_yticks(list(range(len(cancer_to_code))))
ax1.tick_params(axis='y', length=0, width=0, labelsize=8)
ax1.set_xlabel("Five prime", fontsize=12)
ax2.set_xlabel("Ref", fontsize=12)
ax3.set_xlabel("Alt", fontsize=12)
ax4.set_xlabel("Three prime", fontsize=12)
ax1.set_yticklabels(np.array([abbreviations[i] for i in cancer_to_code])[np.argsort([abbreviations[i] for i in cancer_to_code])])
ax2.set_xticks([])
ax3.set_xticks([])
for ax in [ax2, ax3, ax4]:
    ax.set_yticks([])
mpl.colorbar.ColorbarBase(ax5, cmap=myblue, orientation='vertical')
ax5.set_yticks([0, .5, 1])
ax5.set_yticklabels([0, round(vmax * .5, 1), round(vmax, 1)])
ax5.set_title("Bits")

##dels
vmax = np.max([np.max(np.max(five_p_del_information, axis=-1)), np.max(np.max(three_p_del_information, axis=-1)), np.max(np.max(ref_del_information, axis=-1))])
myblue = make_colormap({0: '#ffffff', vmax * .5: '#91a8ee', vmax: '#4169E1'})

fig = plt.figure()
fig.subplots_adjust(left=.056,
                    bottom=.06,
                    right=.95,
                    top=.94,
                    wspace=.1)
gs = fig.add_gridspec(1, 4, width_ratios=[6, 6, 6, .5])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])

ax1.imshow(np.array(five_p_del_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

ax2.imshow(np.array(ref_del_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')


ax3.imshow(np.array(three_p_del_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_label_position('top')
ax1.set_xticks(list(range(6)))
ax1.set_xticklabels([-6, -5, -4, -3, -2, -1])
ax3.set_xticks(list(range(6)))
ax3.set_xticklabels(['+1', '+2', '+3', '+4', '+5', '+6'])
ax1.set_yticks(list(range(len(cancer_to_code))))
ax1.tick_params(axis='y', length=0, width=0, labelsize=8)
ax1.set_xlabel("Five prime", fontsize=12)
ax2.set_xlabel("Ref", fontsize=12)
ax3.set_xlabel("Three prime", fontsize=12)
ax1.set_yticklabels(np.array([abbreviations[i] for i in cancer_to_code])[np.argsort([abbreviations[i] for i in cancer_to_code])])
ax2.set_xticks([])
for ax in [ax2, ax3]:
    ax.set_yticks([])
mpl.colorbar.ColorbarBase(ax4, cmap=myblue, orientation='vertical')
ax4.set_yticks([0, .5, 1])
ax4.set_yticklabels([0, round(vmax * .5, 1), round(vmax, 1)])
ax4.set_title("Bits")


##ins
vmax = np.max([np.max(np.max(five_p_ins_information, axis=-1)), np.max(np.max(three_p_ins_information, axis=-1)), np.max(np.max(alt_ins_information, axis=-1))])
myblue = make_colormap({0: '#ffffff', vmax * .5: '#91a8ee', vmax: '#4169E1'})

fig = plt.figure()
fig.subplots_adjust(left=.056,
                    bottom=.06,
                    right=.95,
                    top=.94,
                    wspace=.1)
gs = fig.add_gridspec(1, 4, width_ratios=[6, 6, 6, .5])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])

ax1.imshow(np.array(five_p_ins_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

ax2.imshow(np.array(alt_ins_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')


ax3.imshow(np.array(three_p_ins_information)[np.argsort([abbreviations[i] for i in cancer_to_code])],
                           cmap=myblue,
                           vmin=0,
                           vmax=vmax,
                           aspect='auto',
                          interpolation='nearest')

for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_label_position('top')
ax1.set_xticks(list(range(6)))
ax1.set_xticklabels([-6, -5, -4, -3, -2, -1])
ax3.set_xticks(list(range(6)))
ax3.set_xticklabels(['+1', '+2', '+3', '+4', '+5', '+6'])
ax1.set_yticks(list(range(len(cancer_to_code))))
ax1.tick_params(axis='y', length=0, width=0, labelsize=8)
ax1.set_xlabel("Five prime", fontsize=12)
ax2.set_xlabel("Alt", fontsize=12)
ax3.set_xlabel("Three prime", fontsize=12)
ax1.set_yticklabels(np.array([abbreviations[i] for i in cancer_to_code])[np.argsort([abbreviations[i] for i in cancer_to_code])])
ax2.set_xticks([])
for ax in [ax2, ax3]:
    ax.set_yticks([])
mpl.colorbar.ColorbarBase(ax4, cmap=myblue, orientation='vertical')
ax4.set_yticks([0, .5, 1])
ax4.set_yticklabels([0, round(vmax * .5, 1), round(vmax, 1)])
ax4.set_title("Bits")