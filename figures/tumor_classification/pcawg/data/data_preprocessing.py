import pickle
import pandas as pd
import numpy as np
import re
from Bio.Seq import Seq
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

tcga_maf = pickle.load(open(cwd / 'files' / 'tcga_pcawg_maf.pkl', 'rb'))
icgc_maf = pickle.load(open(cwd / 'files' / 'icgc_pcawg_maf.pkl', 'rb'))
tcga_maf = pd.concat([tcga_maf, icgc_maf], ignore_index=True)

pcawg_sample_table = pd.read_csv(cwd / 'files' / 'pcawg_sample_sheet.tsv', sep='\t', low_memory=False)

pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['library_strategy'].str.contains('WGS')]
pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['donor_wgs_exclusion_white_gray'] == 'Whitelist']
pcawg_sample_table = pcawg_sample_table.loc[~pcawg_sample_table['dcc_specimen_type'].str.contains('Normal')]
pcawg_sample_table = pcawg_sample_table.loc[~pcawg_sample_table['dcc_specimen_type'].str.contains('Cell line')]

##map to histology
##https://dcc.icgc.org/api/v1/download?fn=/PCAWG/clinical_and_histology/pcawg_specimen_histology_August2016_v9.xlsx
pcawg_histology = pd.read_csv(cwd / 'files' / 'pcawg_specimen_histology_August2016_v9.csv', sep=',', low_memory=False)
histology_mapping = {i: j for i, j in zip(pcawg_histology['icgc_sample_id'], pcawg_histology['histology_abbreviation'])}
pcawg_sample_table['histology'] = pcawg_sample_table['icgc_sample_id'].apply(lambda x: histology_mapping[x])
pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['aliquot_id'].isin(tcga_maf['Tumor_Sample_Barcode'].unique())]
pcawg_sample_table.rename(mapper={'aliquot_id': 'Tumor_Sample_Barcode'}, axis=1, inplace=True)

pcawg_sample_table.reset_index(inplace=True, drop=True)
tcga_maf.reset_index(inplace=True, drop=True)

tcga_maf = pd.merge(tcga_maf, pcawg_sample_table.Tumor_Sample_Barcode.reset_index(), how='left', on='Tumor_Sample_Barcode')
samples_idx = tcga_maf['index'].values

# 5p, 3p, ref, alt
nucleotide_mapping = {'-': 0, 'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
seqs_5p = np.stack(tcga_maf.five_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_3p = np.stack(tcga_maf.three_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_ref = np.stack(tcga_maf.Ref.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_alt = np.stack(tcga_maf.Alt.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)


##need to generate the 96 contexts
def get_context(five_p, three_p, ref, alt, var_type):
    if var_type == 'SNP':
        if ref[0] == 'T' or ref[0] == 'C':
            return five_p[-1] + ref[0] + alt[0] + three_p[0]
        else:
            return str(Seq(three_p).reverse_complement())[-1] + str(Seq(ref[0]).reverse_complement()) + str(Seq(alt[0]).reverse_complement()) + str(Seq(five_p).reverse_complement())[0]
    else:
        return 'none'

variant_types = []
for row in tcga_maf.itertuples():
    if (len(re.findall('[AGCT]', row.Reference_Allele)) == 1 and len(re.findall('[AGCT]', row.Tumor_Seq_Allele2)) == 1):
        variant_types.append('SNP')
    else:
        variant_types.append('other')

tcga_maf['contexts'] = [get_context(five_p, three_p, ref, alt, var_type) for five_p, three_p, ref, alt, var_type in zip(tcga_maf.five_p, tcga_maf.three_p, tcga_maf.Ref, tcga_maf.Alt, variant_types)]

instances = {'sample_idx': samples_idx,
             'seq_5p': seqs_5p,
             'seq_3p': seqs_3p,
             'seq_ref': seqs_ref,
             'seq_alt': seqs_alt,
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

##set strand to 0 since we don't have information about strand for WGS
instances['strand'] = np.repeat(0, len(tcga_maf))

instances, tcga_maf, pcawg_sample_table = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'pcawg' / 'data' / 'data.pkl', 'rb'))
tcga_maf.drop(labels=['bin'], axis=1, inplace=True)
chromosomes = {}
for i in list(range(1, 23))+['X', 'Y']:
    with open(cwd / 'files' / 'chromosomes' / ('chr' + str(i) + '.txt')) as f:
        chromosomes[str(i)] = f.read()

##need to generate genomic bins
def get_bin(chromosome, position):
    global_pos = 0
    for i in list(range(1, 23))+['X', 'Y']:
        if str(chromosome) == str(i):
            break
        global_pos += len(chromosomes[str(i)])
    global_pos += position
    return global_pos

tcga_maf['genome_position'] = [get_bin(chromosome, position) for chromosome, position in zip(tcga_maf.Chromosome, tcga_maf.Start_position)]

with open(cwd / 'figures' / 'tumor_classification' / 'pcawg' / 'data' / 'data.pkl', 'wb') as f:
    pickle.dump([instances, tcga_maf, pcawg_sample_table], f)
