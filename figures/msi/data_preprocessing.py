import pickle
import pandas as pd
import numpy as np
import re
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

##path to files
path = cwd / 'files/'

tcga_maf = pickle.load(open(path / 'tcga_maf_table_controlled.pkl', 'rb'))
samples = pickle.load(open(path / 'tcga_sample_table_controlled.pkl', 'rb'))
panels = pickle.load(open(path / 'tcga_panel_table_controlled.pkl', 'rb'))
samples = samples.loc[~pd.isna(samples['msi'])]
samples.reset_index(inplace=True, drop=True)

##some samples don't have a cancer label
import json
with open(path / 'cases.2020-02-28.json', 'r') as f:
    tcga_cancer_info = json.load(f)
cancer_labels = {i['submitter_id']: i['project']['project_id'].split('-')[-1] for i in tcga_cancer_info}

samples['type'] = samples['bcr_patient_barcode'].apply(lambda x: cancer_labels[x])

tcga_maf = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'].isin(samples.Tumor_Sample_Barcode.values)]

##there's some nans
tcga_maf.dropna(inplace=True, subset=['Ref'])
tcga_maf.reset_index(inplace=True, drop=True)

##create a new column called index that is the sample idxs
tcga_maf = pd.merge(tcga_maf, samples.Tumor_Sample_Barcode.reset_index(), how='left', on='Tumor_Sample_Barcode')

##if you want to check indexes match up
# maf_indexes = {i: j for i, j in zip(tcga_maf['index'].values, tcga_maf['Tumor_Sample_Barcode'].values)}
# sample_indexes = {i: j for i, j in zip(samples.index.values, samples['Tumor_Sample_Barcode'].values)}
# X = True
# for index in maf_indexes:
#     if maf_indexes[index] != sample_indexes[index]:
#         X=False
# print(X)

samples_idx = tcga_maf['index'].values

# 5p, 3p, ref, alt
nucleotide_mapping = {'-': 0, 'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
seqs_5p = np.stack(tcga_maf.five_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_3p = np.stack(tcga_maf.three_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_ref = np.stack(tcga_maf.Ref.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_alt = np.stack(tcga_maf.Alt.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)

cds = tcga_maf['CDS_position'].astype(str).apply(lambda x: (int(x) % 3) + 1 if re.match('^[0-9]+$', x) else 0).values

instances = {'sample_idx': samples_idx,
             'seq_5p': seqs_5p,
             'seq_3p': seqs_3p,
             'seq_ref': seqs_ref,
             'seq_alt': seqs_alt,
             'cds': cds,
             'repeat': tcga_maf['repeat'].values}

msi_dict = {'MSS': 'low', 'MSI-L': 'low', 'MSI-H': 'high'}
samples['msi_status'] = samples.msi.apply(lambda x: msi_dict[x])

A = samples.msi_status.astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]

samples_dict = {'cancer': samples.type.values,
           'histology': samples.histological_type.values,
           'mantis': samples['MANTIS Score'].values,
           'age': samples.age_at_initial_pathologic_diagnosis.values,
           'class': classes_onehot,
           'classes': classes}

variant_encoding = np.array([0, 2, 1, 4, 3])

##for stacked
instances['seq_5p'] = np.stack([instances['seq_5p'], variant_encoding[instances['seq_3p'][:, ::-1]]], axis=2)
instances['seq_3p'] = np.stack([instances['seq_3p'], variant_encoding[instances['seq_5p'][:, :, 0][:, ::-1]]], axis=2)
t = instances['seq_ref'].copy()
i = t != 0
t[i] = variant_encoding[instances['seq_ref'][:, ::-1]][i[:, ::-1]]
instances['seq_ref'] = np.stack([instances['seq_ref'], t], axis=2)
t = instances['seq_alt'].copy()
i = t != 0
t[i] = variant_encoding[instances['seq_alt'][:, ::-1]][i[:, ::-1]]
instances['seq_alt'] = np.stack([instances['seq_alt'], t], axis=2)
del i, t

instances['strand'] = tcga_maf['STRAND'].astype(str).apply(lambda x: {'.': 0, '-1': 1, '1': 2}[x]).values

with open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'wb') as f:
    pickle.dump([instances, samples_dict, samples], f)
