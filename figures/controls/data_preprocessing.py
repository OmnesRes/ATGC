import pickle
import pandas as pd
import numpy as np
import re
from Bio.Seq import Seq
import pyranges as pr
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]


tcga_maf = pickle.load(open(cwd / 'files' / 'tcga_maf_table.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_sample_table.pkl', 'rb'))
##there's a bad sample with nans
tcga_maf.dropna(inplace=True)
tcga_maf.reset_index(inplace=True, drop=True)
tcga_maf = pd.merge(tcga_maf, samples.Tumor_Sample_Barcode.reset_index(), how='left', on='Tumor_Sample_Barcode')
samples_idx = tcga_maf['index'].values


# 5p, 3p, ref, alt
nucleotide_mapping = {'-': 0, 'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
seqs_5p = np.stack(tcga_maf.five_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_3p = np.stack(tcga_maf.three_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_ref = np.stack(tcga_maf.Ref.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_alt = np.stack(tcga_maf.Alt.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)

##map genomic coordinates to the exome (takes some time)

# chr, pos
chromosome_mapping = dict(zip([str(i) for i in list(range(1, 23))] + ['X', 'Y'], list(range(1, 25))))
gen_chr = np.array([chromosome_mapping[i] for i in tcga_maf.Chromosome.values])

gff = pd.read_csv(cwd / 'files'/ 'Homo_sapiens.GRCh37.87.gff3',
                  sep='\t',
                  names=['chr', 'unknown', 'gene_part', 'start', 'end', 'unknown2', 'strand', 'unknown3', 'gene_info'],
                  usecols=['chr','gene_part', 'start', 'end', 'gene_info'],
                  low_memory=False)

gff_cds_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'CDS') & gff['chr'].isin(tcga_maf['Chromosome'].unique()), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge().sort()

gff_dfs = gff_cds_pr.dfs

for df in gff_dfs:
    gff_dfs[df]['length'] = gff_dfs[df]['End'] - gff_dfs[df]['Start']

cum_pos = []
for row in tcga_maf.itertuples():
    cum_pos.append(gff_dfs[row.Chromosome][gff_dfs[row.Chromosome]['Start'] < row.Start_Position]['length'].sum())

chromosome_cds_sizes = {df: gff_dfs[df]['length'].sum() for df in gff_dfs}

gen_pos = np.array(cum_pos) / [chromosome_cds_sizes[i] for i in tcga_maf.Chromosome.values]


def get_cds(cds):
    if '-' in cds:
        cds = cds.split('-')[0]
    if re.match('^[0-9]+$', cds):
        return (int(cds) % 3) + 1
    else:
        return 0

cds = tcga_maf['CDS_position'].astype(str).apply(get_cds)

##need to generate the 96 contexts
def get_context(five_p, three_p, ref, alt, var_type):
    if var_type == 'SNP':
        if ref[0] == 'T' or ref[0] == 'C':
            return five_p[-1] + ref[0] + alt[0] + three_p[0]
        else:
            return str(Seq(three_p).reverse_complement())[-1] + str(Seq(ref[0]).reverse_complement()) + str(Seq(alt[0]).reverse_complement()) + str(Seq(five_p).reverse_complement())[0]
    else:
        return 'none'

tcga_maf['contexts'] = [get_context(five_p, three_p, ref, alt, var_type) for five_p, three_p, ref, alt, var_type in zip(tcga_maf.five_p, tcga_maf.three_p, tcga_maf.Ref, tcga_maf.Alt, tcga_maf.Variant_Type)]

instances = {'sample_idx': samples_idx,
             'seq_5p': seqs_5p,
             'seq_3p': seqs_3p,
             'seq_ref': seqs_ref,
             'seq_alt': seqs_alt,
             'chr': gen_chr,
             'pos_float': gen_pos,
             'cds': cds,
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

with open(cwd / 'figures' / 'controls' / 'data' / 'data.pkl', 'wb') as f:
    pickle.dump([instances, tcga_maf], f)