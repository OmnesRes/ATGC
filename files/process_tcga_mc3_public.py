import pandas as pd
import numpy as np
import pickle
import pyranges as pr
import concurrent.futures
import pathlib
from tqdm import tqdm
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
##your path to the files directory
file_path = cwd / 'files/'

# load tcga clinical file  download from 'https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81'
tcga_sample_table = pd.read_csv(file_path / 'TCGA-CDR-SupplementalTableS1.tsv', sep='\t').iloc[:, 1:]
tcga_sample_table['histological_type'].fillna('', inplace=True)



# load pathology annotations from asb
tumor_types = pd.read_csv(file_path / 'tumor_types_NCI-T.csv', sep='\t')
tumor_types.fillna('', inplace=True)
tcga_sample_table = pd.merge(tcga_sample_table, tumor_types[['type', 'histological_type', 'NCI-T Label', 'NCI-T Code']], how='left', on=['type', 'histological_type'])

# tcga mc3 filename download from 'https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc'
mc3_file_name = file_path / 'mc3.v0.2.8.PUBLIC.maf'
# these are a core set of basic columns we would want
usecols = ['Hugo_Symbol', 'Hugo_Symbol', 'Center', 'NCBI_Build', 'Chromosome', 'Start_Position', 'End_Position', 'STRAND', 'Variant_Classification', 'Variant_Type', 'Consequence', 'Reference_Allele', 'Tumor_Seq_Allele2', 't_ref_count', 't_alt_count', 'Tumor_Sample_Barcode', 'CONTEXT', 'FILTER', 'CDS_position']
tcga_maf = pd.read_csv(mc3_file_name, sep='\t', usecols=usecols, low_memory=False)
##The MAF contains nonpreferred pairs which results in some samples having duplicated variants
tcga_maf = tcga_maf.loc[(tcga_maf['FILTER'] == 'PASS') | (tcga_maf['FILTER'] == 'wga') | (tcga_maf['FILTER'] == 'native_wga_mix')]

##chang hotspots
hotspots = pd.read_csv(cwd / 'files' / 'hotspots.vcf', skiprows=1, sep='\t')
hotspots = hotspots[['#CHROM', 'POS', 'REF', 'ALT']]
hotspots.drop_duplicates(inplace=True)


def get_overlap(tumor):
    tumor_df = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'] == tumor].sort_values(['Start_Position'])
    ##merge sequential SNPs into a single mutation
    dfs = []
    problems = []
    for i in tumor_df['Chromosome'].unique():
        result = tumor_df.loc[(tumor_df['Chromosome'] == i) & (tumor_df['Variant_Type'] == 'SNP')].copy()
        if len(result) > 1:
            to_merge = sum(result['Start_Position'].values - result['Start_Position'].values[:, np.newaxis] == 1)
            merged = []
            position = 0
            indexes_to_remove = []
            while sum(to_merge[position:]) > 0 and position < len(to_merge) - 1:
                for index, merge in enumerate(to_merge[position:]):
                    if merge:
                        first = position + index - 1
                        last = position + index
                        while to_merge[last]:
                            last += 1
                            if last < len(to_merge):
                                pass
                            else:
                                break
                        position = last
                        last -= 1
                        ref = ''
                        alt = ''
                        variant = result.iloc[[first]].copy()
                        variant['End_Position'] = result.iloc[last]['Start_Position']
                        variant['Variant_Classification'] = 'Missense_Mutation'
                        variant['Variant_Classification']
                        if last - first == 1:
                            type = 'DNP'
                        elif last - first == 2:
                            type = 'TNP'
                        else:
                            type = 'ONP'
                        variant['Variant_Type'] = type
                        variant['Consequence'] = 'missense_variant'
                        variant['CONTEXT'] = False
                        alt_counts = []
                        ref_counts = []
                        for row in result.iloc[first:last + 1, :].itertuples():
                            ref += row.Reference_Allele
                            alt += row.Tumor_Seq_Allele2
                            ref_counts.append(row.t_ref_count)
                            alt_counts.append(row.t_alt_count)
                        variant['t_ref_count'] = min(ref_counts)
                        variant['t_alt_count'] = min(alt_counts)
                        variant['Reference_Allele'] = ref
                        variant['Tumor_Seq_Allele2'] = alt
                        ##decide whether or not to merge
                        mean_vaf = np.mean([alt / (ref + alt) for ref, alt in zip(ref_counts, alt_counts)])
                        vaf_deviation = max([np.abs(mean_vaf - (alt / (ref + alt))) / mean_vaf for ref, alt in zip(ref_counts, alt_counts)])
                        ref_mean = max(np.mean(ref_counts), .00001)
                        ref_deviation_percent = max([np.abs(ref_mean - ref) / ref_mean for ref in ref_counts])
                        ref_deviation =max([np.abs(ref_mean - ref) for ref in ref_counts])
                        alt_mean = np.mean(alt_counts)
                        alt_deviation_percent = max([np.abs(alt_mean - alt) / alt_mean for alt in alt_counts])
                        alt_deviation = max([np.abs(alt_mean - alt) for alt in alt_counts])
                        if vaf_deviation < .05 or alt_deviation_percent < .05 or ref_deviation_percent < .05 or alt_deviation < 5 or ref_deviation < 5:
                            indexes_to_remove += list(range(first, last + 1))
                            merged.append(variant)
                        break
            result = result[~np.array([i in indexes_to_remove for i in range(len(result))])]
            if len(result) > 0 and len(merged) > 0:
                result = pd.concat([result, pd.concat(merged, ignore_index=True)], ignore_index=True)
            elif len(merged) == 0:
                pass
            else:
                result = pd.concat(merged, ignore_index=True)
            dfs.append(result)
        else:
            dfs.append(result)
    tumor_df = pd.concat([pd.concat(dfs, ignore_index=True), tumor_df.loc[tumor_df['Variant_Type'] != 'SNP'].copy()], ignore_index=True)
    return tumor_df

# data = {}
# with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
#     for tumor, result in tqdm(zip(tumor_to_bed.keys(), executor.map(get_overlap, tumor_to_bed.keys()))):
#         data[tumor] = result

data = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
    for tumor, result in tqdm(zip(tcga_maf['Tumor_Sample_Barcode'].unique(), executor.map(get_overlap, tcga_maf['Tumor_Sample_Barcode'].unique()))):
        data[tumor] = result


##need to merge chr 1, 12725999, 12726000, TCGA-FW-A3R5-06A-11D-A23B-08




##do the bed intersections here, get sizes for sample table
# tcga_sample_table['exome'] = tcga_sample_table['Tumor_Sample_Barcode'].apply(lambda x: sizes(x))



# df of counts via groupby, could add other metrics derived from mc maf here
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
tcga_counts = tcga_maf[['Variant_Classification', 'Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x), (x['Variant_Classification'].isin(non_syn)).sum()], index=['all_counts', 'non_syn_counts']))
tcga_counts['non_syn_tmb'] = tcga_counts['non_syn_counts'] / 31.85
tcga_counts.reset_index(inplace=True)
# linkage to pancan clinical annotation table via sample barcode
tcga_counts['bcr_patient_barcode'] = tcga_counts['Tumor_Sample_Barcode'].str.extract(r'^([^-]+-[^-]+-[^-]+)-')
tcga_counts['bcr_sample_barcode'] = tcga_counts['Tumor_Sample_Barcode'].str.extract(r'^([^-]+-[^-]+-[^-]+-[^-]+)-')

# join to clinical annotation for data in mc3 only, this will add Tumor_Sample_Barcode also to the tcga_sample_table
tcga_sample_table = pd.merge(tcga_sample_table, tcga_counts, how='right', on='bcr_patient_barcode')

# add MANTIS data downloaded from 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5972025/bin/NIHMS962713-supplement-File_S2.xlsx'
mantis_file_name = file_path / 'NIHMS962713-supplement-File_S2.csv'
mantis_df = pd.read_csv(mantis_file_name, sep='\t')
mantis_df['Tumor_Sample_Barcode'] = mantis_df['Tumor Filename'].str.extract(r'(TCGA[^\.]+)\.')
tcga_sample_table = pd.merge(tcga_sample_table, mantis_df[['Tumor_Sample_Barcode', 'MANTIS Score']], how='left', on='Tumor_Sample_Barcode')

#add msi pcr labels, data processing and data sources available at ATGC/files/msi_ground_truth/
msi = pickle.load(open(file_path / 'msi_ground_truth' / 'msi_labels.pkl', 'rb'))
tcga_sample_table = pd.merge(tcga_sample_table, msi, how='left', left_on='bcr_patient_barcode', right_index=True)
tcga_sample_table.rename(columns={0: 'msi'}, inplace=True)


##sample table is done, save to file
pickle.dump(tcga_sample_table, open(file_path / 'tcga_sample_table.pkl', 'wb'))


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
tcga_maf['index'] = tcga_maf.index.values

maf_pr = pr.PyRanges(tcga_maf.loc[:, ['Chromosome', 'Start_Position', 'End_Position', 'index']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))

##use the genie 7.0 panels: https://www.synapse.org/#!Synapse:syn21551261
genie = pd.read_csv(file_path / 'genomic_information.txt', sep='\t', low_memory=False)
panels = genie.SEQ_ASSAY_ID.unique()
panel_df = pd.DataFrame(data=panels, columns=['Panel'])


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


grs = {k: v for k, v in zip(['CDS', 'exon'] + list(panels), [gff_cds_pr, gff_exon_pr] + panel_prs)}
result = pr.count_overlaps(grs, pr.concat({'maf': maf_pr}.values()))
result = result.df

tcga_maf = pd.merge(tcga_maf, result.iloc[:, 3:], how='left', on='index')


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
        Start = row.Start_Position
        End = row.End_Position
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

tcga_maf['Ref'], tcga_maf['Alt'], tcga_maf['five_p'], tcga_maf['three_p'] = variant_features(tcga_maf)

tcga_maf.drop(columns=['index'], inplace=True)

pickle.dump(tcga_maf, open(file_path / 'tcga_maf_table.pkl', 'wb'))
pickle.dump(panel_df, open(file_path / 'tcga_panel_table.pkl', 'wb'))

