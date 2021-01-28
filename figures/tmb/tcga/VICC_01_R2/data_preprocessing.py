import pickle
import pandas as pd
import numpy as np
import re
import json
import pathlib

path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]

##path to files
path = cwd / 'files/'

tcga_maf = pickle.load(open(path / 'tcga_maf_table.pkl', 'rb'))
samples = pickle.load(open(path / 'tcga_sample_table.pkl', 'rb'))
panels = pickle.load(open(path / 'tcga_panel_table.pkl', 'rb'))
#
##fill in some missing cancer labels
with open(path / 'cases.2020-02-28.json', 'r') as f:
    tcga_cancer_info = json.load(f)
cancer_labels = {i['submitter_id']: i['project']['project_id'].split('-')[-1] for i in tcga_cancer_info}
cancer_labels['TCGA-AB-2852'] = 'LAML'
samples['type'] = samples['bcr_patient_barcode'].apply(lambda x: cancer_labels[x])

##remove samples without a kit that covered the exome
samples_covered = samples.loc[samples['Exome_Covered']]
samples_unknown = samples.loc[(samples['Exome_Unknown']) & (samples['type'].isin(['KIRC', 'BRCA']))]
samples = samples_covered.append(samples_unknown)

##remove samples with TMB above 64
samples = samples.loc[samples['non_syn_counts'] / (panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) < 64]

samples.reset_index(inplace=True, drop=True)

##limit MAF to samples
tcga_maf = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'].isin(samples.Tumor_Sample_Barcode.values)]

##for limiting to oncopanel
tcga_maf = tcga_maf.loc[(tcga_maf['VICC-01-R2'] > 0) & (tcga_maf['CDS'] > 0)]
tcga_maf.reset_index(inplace=True, drop=True)

##create a new column called index that is the sample idxs
tcga_maf = pd.merge(tcga_maf, samples.Tumor_Sample_Barcode.reset_index(), how='left', on='Tumor_Sample_Barcode')

##if you want to check indexes match up
maf_indexes = {i: j for i, j in zip(tcga_maf['index'].values, tcga_maf['Tumor_Sample_Barcode'].values)}
sample_indexes = {i: j for i, j in zip(samples.index.values, samples['Tumor_Sample_Barcode'].values)}
X = True
for index in maf_indexes:
    if maf_indexes[index] != sample_indexes[index]:
        X = False
print(X)

samples_idx = tcga_maf['index'].values

# 5p, 3p, ref, alt
nucleotide_mapping = {'-': 0, 'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
seqs_5p = np.stack(tcga_maf.five_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x[-6:]])).values, axis=0)
seqs_3p = np.stack(tcga_maf.three_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x[:6]])).values, axis=0)
seqs_ref = np.stack(tcga_maf.Ref.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_alt = np.stack(tcga_maf.Alt.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)

# chr, pos
chromosome_mapping = dict(zip([str(i) for i in list(range(1, 23))] + ['X', 'Y'], list(range(1, 25))))
gen_chr = np.array([chromosome_mapping[i] for i in tcga_maf.Chromosome.values])

with open(path + 'chr_sizes.tsv') as f:
    sizes = [i.split('\t') for i in f.read().split('\n')[:-1]]

chromosome_sizes = {i: float(j) for i, j in sizes}
gen_pos = tcga_maf['Start_Position'].values / [chromosome_sizes[i] for i in tcga_maf.Chromosome.values]

cds = tcga_maf['CDS_position'].astype(str).apply(lambda x: (int(x) % 3) + 1 if re.match('^[0-9]+$', x) else 0).values

instances = {'sample_idx': samples_idx,
             'seq_5p': seqs_5p,
             'seq_3p': seqs_3p,
             'seq_ref': seqs_ref,
             'seq_alt': seqs_alt,
             'chr': gen_chr,
             'pos_float': gen_pos,
             'cds': cds}

##histology embedding
A = samples['type'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]

samples_dict = {'cancer': samples.type.values,
           'histology': classes_onehot,
           'histology_labels': classes,
           'mantis': samples['MANTIS Score'].values,
           'age': samples.age_at_initial_pathologic_diagnosis.values,
           }

variant_encoding = np.array([0, 2, 1, 4, 3])
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

with open(cwd / 'figures' / 'tmb' / 'tcga' / 'VICC_01_R2' / 'data' / 'data.pkl', 'wb') as f:
    pickle.dump([instances, samples_dict, tcga_maf, samples], f)

