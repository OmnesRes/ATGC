import pandas as pd
import numpy as np
import pickle
import pyranges as pr
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
##your path to the files directory
file_path = cwd / 'files/'

usecols = ['Hugo_Symbol', 'Chromosome', 'Start_position', 'End_position', 'Variant_Classification', 'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele2', 'i_VAF', 'Tumor_Sample_Barcode', 'Donor_ID']

##from: https://dcc.icgc.org/releases/PCAWG/consensus_snv_indel
pcawg_maf = pd.read_csv(file_path / 'final_consensus_passonly.snv_mnv_indel.icgc.public.maf', sep='\t',
                        usecols=usecols,
                        low_memory=False)


##from: https://dcc.icgc.org/releases/PCAWG/donors_and_biospecimens
pcawg_sample_table = pd.read_csv(file_path / 'pcawg_sample_sheet.tsv', sep='\t', low_memory=False)
##limit samples to what's in the maf
pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['aliquot_id'].isin(pcawg_maf['Tumor_Sample_Barcode'].unique())]
pcawg_sample_table.drop_duplicates(['icgc_donor_id'], inplace=True)
pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['dcc_specimen_type'] != 'Cell line - derived from tumour']
##from: https://dcc.icgc.org/releases/current/Summary
pcawg_donor_table = pd.read_csv(file_path / 'donor.all_projects.tsv', sep='\t', low_memory=False)
pcawg_sample_table = pd.merge(pcawg_sample_table, pcawg_donor_table, how='left', on='icgc_donor_id')


##limit MAF to unique samples
pcawg_maf = pcawg_maf.loc[pcawg_maf['Tumor_Sample_Barcode'].isin(pcawg_sample_table['aliquot_id'])]


# df of counts via groupby, could add other metrics derived from mc maf here
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
pcawg_counts = pcawg_maf[['Variant_Classification', 'Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x), (x['Variant_Classification'].isin(non_syn)).sum()], index=['all_counts', 'non_syn_counts']))
pcawg_counts['non_syn_tmb'] = pcawg_counts['non_syn_counts'] / 31.85
pcawg_counts.reset_index(inplace=True)

# join to clinical annotation for data in mc3 only, this will add Tumor_Sample_Barcode also to the tcga_sample_table
pcawg_sample_table = pd.merge(pcawg_sample_table, pcawg_counts, how='right', left_on='aliquot_id', right_on='Tumor_Sample_Barcode')


##sample table is done, save to file
pickle.dump(pcawg_sample_table, open(file_path / 'pcawg_sample_table.pkl', 'wb'))

chromosomes = {}
for i in list(range(1, 23))+['X', 'Y']:
    with open(file_path / 'chromosomes' / ('chr' + str(i) + '.txt')) as f:
        chromosomes[str(i)] = f.read()


##Use GFF3 to annotate variants
##ftp://ftp.ensembl.org/pub/grch37/current/gff3/homo_sapiens/
gff = pd.read_csv(file_path / 'Homo_sapiens.GRCh37.87.gff3',
                  sep='\t',
                  names=['chr', 'unknown', 'gene_part', 'start', 'end', 'unknown2', 'strand', 'unknown3', 'gene_info'],
                  usecols=['chr','gene_part', 'start', 'end', 'gene_info'],
                  low_memory=False)


gff_cds_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'CDS') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()
gff_exon_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'exon') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()
del gff

##make index column for merging
pcawg_maf['index'] = pcawg_maf.index.values

maf_pr = pr.PyRanges(pcawg_maf.loc[:, ['Chromosome', 'Start_position', 'End_position', 'index']].rename(columns={'Start_position': 'Start', 'End_position': 'End'}))

##use the genie 7.0 panels: https://www.synapse.org/#!Synapse:syn21551261
genie = pd.read_csv(file_path / 'genomic_information.txt', sep='\t', low_memory=False)
panels = genie.SEQ_ASSAY_ID.unique()
panel_df = pd.DataFrame(data=panels, columns=['Panel'])

repeats = pd.read_csv(file_path / 'simpleRepeat.txt', sep='\t', low_memory=False, header=None, usecols=[1, 2, 3])
repeats[1] = repeats[1].str.replace('chr', '')
repeats.rename(columns={1: 'Chromosome', 2: 'Start', 3: 'End'}, inplace=True)
repeats_pr = pr.PyRanges(repeats.loc[repeats['Chromosome'].isin(chromosomes)]).merge()

total_sizes = []
cds_sizes = []
exon_sizes = []
panel_prs = []

for panel in panels:
    print(panel)
    panel_pr = pr.PyRanges(genie.loc[(genie['SEQ_ASSAY_ID'] == panel) & genie['Chromosome'].isin(chromosomes), 'Chromosome':'End_Position'].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'})).merge()
    total_sizes.append(sum([i + 1 for i in panel_pr.lengths()]))
    cds_sizes.append(sum([i + 1 for i in panel_pr.intersect(gff_cds_pr).lengths()]))
    exon_sizes.append(sum([i + 1 for i in panel_pr.intersect(gff_exon_pr).lengths()]))
    panel_prs.append(panel_pr)


grs = {k: v for k, v in zip(['repeat', 'CDS', 'exon'] + list(panels), [repeats_pr, gff_cds_pr, gff_exon_pr] + panel_prs)}
result = pr.count_overlaps(grs, pr.concat({'maf': maf_pr}.values()))
result = result.df

pcawg_maf = pd.merge(pcawg_maf, result.iloc[:, 3:], how='left', on='index')

panel_df['total'] = total_sizes
panel_df['cds'] = cds_sizes
panel_df['exon'] = exon_sizes

##get assumed size of the most common kit: https://bitbucket.org/cghub/cghub-capture-kit-info/src/master/BI/vendor/Agilent/whole_exome_agilent_1.1_refseq_plus_3_boosters.targetIntervals.bed
agilent_df = pd.read_csv(file_path / 'whole_exome_agilent_1.1_refseq_plus_3_boosters.targetIntervals.bed', sep='\t', low_memory=False, header=None)
kit_pr = pr.PyRanges(agilent_df.rename(columns={0: 'Chromosome', 1: 'Start', 2: 'End'})).merge()
kit_total = sum([i + 1 for i in kit_pr.lengths()])
kit_cds = sum([i + 1 for i in kit_pr.intersect(gff_cds_pr).merge().lengths()])
kit_exon = sum([i + 1 for i in kit_pr.intersect(gff_exon_pr).merge().lengths()])

panel_df = panel_df.append({'Panel': 'Agilent_kit', 'total': kit_total, 'cds': kit_cds, 'exon': kit_exon}, ignore_index=True)


def variant_features(maf, ref_length=6, alt_length=6, five_p_length=6, three_p_length=6):
    refs = []
    alts = []
    five_ps = []
    three_ps = []
    if ref_length % 2 != 0:
        ref_length += 1
        print('Your ref length was not even, incrementing by 1.')
    if alt_length % 2 != 0:
        alt_length += 1
        print('Your alt length was not even, incrementing by 1.')

    for index, row in enumerate(maf.itertuples()):
        Ref = row.Reference_Allele
        Alt = row.Tumor_Seq_Allele2
        Chr = str(row.Chromosome)
        Start = row.Start_position
        End = row.End_position
        if pd.isna(Alt):
            print(str(index)+' Alt is nan')
            Ref = np.nan
            Alt = np.nan
            context_5p = np.nan
            context_3p = np.nan
        else:
            if len(Ref) > ref_length:
                Ref = Ref[:int(ref_length / 2)] + Ref[-int(ref_length / 2):]
            else:
                while len(Ref) < ref_length:
                    Ref += '-'
            if len(Alt) > alt_length:
                Alt = Alt[:int(alt_length / 2)] + Alt[-int(alt_length / 2):]
            else:
                while len(Alt) < alt_length:
                    Alt += '-'
            if row.Reference_Allele == '-':
                ##the TCGA coordinates for a null ref are a little weird
                assert Start-five_p_length >= 0
                context_5p = chromosomes[Chr][Start-five_p_length:Start]
                context_3p = chromosomes[Chr][Start:Start+three_p_length]
            else:
                assert Start-(five_p_length+1) >= 0
                context_5p = chromosomes[Chr][Start-(five_p_length+1):Start-1]
                context_3p = chromosomes[Chr][End:End+three_p_length]
        refs.append(Ref)
        alts.append(Alt)
        five_ps.append(context_5p)
        three_ps.append(context_3p)
    return refs, alts, five_ps, three_ps

pcawg_maf['Ref'], pcawg_maf['Alt'], pcawg_maf['five_p'], pcawg_maf['three_p'] = variant_features(pcawg_maf)

pcawg_maf.drop(columns=['index'], inplace=True)

pickle.dump(pcawg_maf, open(file_path / 'pcawg_maf_table.pkl', 'wb'), protocol=4)
pickle.dump(panel_df, open(file_path / 'pcawg_panel_table.pkl', 'wb')) ##should be same as tcga_panel_table

