import pickle
import pandas as pd
import numpy as np
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]

##path to files
path = cwd / 'files/'


tcga_maf = pickle.load(open(path / 'pcawg_maf_table.pkl', 'rb'))
samples = pickle.load(open(path / 'pcawg_sample_table.pkl', 'rb'))
panels = pickle.load(open(path / 'pcawg_panel_table.pkl', 'rb'))


##remove samples with TMB above 64
samples = samples.loc[samples['non_syn_counts'] / (panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) < 64]

samples.reset_index(inplace=True, drop=True)

##limit MAF to samples
tcga_maf = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'].isin(samples.Tumor_Sample_Barcode.values)]

##for limiting to oncopanel
tcga_maf = tcga_maf.loc[tcga_maf['DFCI-ONCOPANEL-3'] > 0]
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
seqs_5p = np.stack(tcga_maf.five_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_3p = np.stack(tcga_maf.three_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_ref = np.stack(tcga_maf.Ref.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_alt = np.stack(tcga_maf.Alt.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)


# chr, pos
chromosome_mapping = dict(zip([str(i) for i in list(range(1, 23))] + ['X', 'Y'], list(range(1, 25))))
gen_chr = np.array([chromosome_mapping[i] for i in tcga_maf.Chromosome.values])

with open(path / 'chr_sizes.tsv') as f:
    sizes = [i.split('\t') for i in f.read().split('\n')[:-1]]

chromosome_sizes = {i: float(j) for i, j in sizes}
gen_pos = tcga_maf['Start_position'].values / [chromosome_sizes[i] for i in tcga_maf.Chromosome.values]

cds = np.ones(len(gen_pos))

instances = {'sample_idx': samples_idx,
             'seq_5p': seqs_5p,
             'seq_3p': seqs_3p,
             'seq_ref': seqs_ref,
             'seq_alt': seqs_alt,
             'chr': gen_chr,
             'pos_float': gen_pos,
             'cds': cds}

##histology embedding
samples['cancer_code'] = samples['project_code'].apply(lambda x: x.split('-')[0])
cancer_dict = {'LICA': 'liver', 'LINC': 'liver', 'LIRI': 'liver', 'BTCA': 'liver', 'BOCA': 'bone', 'BRCA': 'breast',
               'CLLE': 'blood', 'CMDI': 'blood', 'LAML': 'blood', 'MALY': 'blood', 'EOPC': 'prostate', 'PRAD': 'prostate',
               'OV': 'ovarian', 'MELA': 'skin', 'ESAD':'orogastric', 'ORCA': 'orogastric', 'GACA': 'orogastric',
               'PBCA': 'brain', 'RECA': 'renal', 'PAEN': 'pancreas', 'PACA': 'pancreas'}

samples['cancer'] = samples['cancer_code'].apply(lambda x: cancer_dict[x])
A = samples['cancer'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]

samples_dict = {'histology': classes_onehot,
           'histology_labels': classes,
           'age': samples.donor_age_at_diagnosis.values,
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

instances['strand'] = np.ones((tcga_maf.shape[0], ))

with open(cwd / 'figures' / 'tmb' / 'pcawg' / 'DFCI_ONCO' / 'data' / 'data.pkl', 'wb') as f:
    pickle.dump([instances, samples_dict, tcga_maf, samples], f)

