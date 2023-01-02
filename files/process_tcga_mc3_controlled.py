import pandas as pd
import numpy as np
import pickle
import pyranges as pr
import json
import subprocess
import concurrent.futures
import pathlib
from tqdm import tqdm
import re
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
##your path to the files directory
file_path = cwd / 'files/'

# tcga mc3 filename download from 'https://gdc.cancer.gov/about-data/publications/mc3-2017'
mc3_file_name = 'mc3.v0.2.8.CONTROLLED.maf'
# these are a core set of basic columns we would want
usecols = ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'STRAND', 'Variant_Classification', 'Variant_Type', 'Gene', 'Reference_Allele', 'Tumor_Seq_Allele2', 't_ref_count', 't_alt_count', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode', 'FILTER', 'CDS_position', 'NCALLERS']
tcga_maf = pd.read_csv(file_path / mc3_file_name, sep='\t', usecols=usecols, low_memory=False)

##The MAF contains nonpreferred pairs which results in some samples having duplicated variants
filters = ['PASS', 'NonExonic,bitgt', 'NonExonic,bitgt,wga', 'NonExonic', 'NonExonic,wga', 'bitgt', 'bitgt,wga', 'wga', \
           'broad_PoN_v2', 'NonExonic,bitgt,broad_PoN_v2', 'NonExonic,bitgt,broad_PoN_v2,wga', 'NonExonic,broad_PoN_v2', \
           'broad_PoN_v2,wga', 'bitgt,broad_PoN_v2', 'NonExonic,broad_PoN_v2,wga', 'bitgt,broad_PoN_v2,wga', \
           'NonExonic,bitgt,native_wga_mix', 'NonExonic,native_wga_mix', 'bitgt,native_wga_mix', 'native_wga_mix', \
           'NonExonic,bitgt,broad_PoN_v2,native_wga_mix', 'broad_PoN_v2,native_wga_mix', 'NonExonic,broad_PoN_v2,native_wga_mix', \
           'bitgt,broad_PoN_v2,native_wga_mix']

tcga_maf = tcga_maf.loc[tcga_maf['FILTER'].isin(filters)]
tcga_maf = tcga_maf.loc[tcga_maf['Chromosome'] != 'MT']
tcga_maf = tcga_maf.loc[tcga_maf['NCALLERS'] > 1]

chromosomes = {}
for i in list(range(1, 23))+['X', 'Y']:
    with open(file_path / 'chromosomes' / ('chr' + str(i) + '.txt')) as f:
        chromosomes[str(i)] = f.read()

##there's a TNP that should only be merged into a DNP, remove the SNP then add it back
##need to merge chr 1, 12725999, 12726000, TCGA-FW-A3R5-06A-11D-A23B-08
temp = tcga_maf.loc[(tcga_maf['Tumor_Sample_Barcode'] == 'TCGA-FW-A3R5-06A-11D-A23B-08') &\
                    (tcga_maf['Chromosome'] == '1') & (tcga_maf['Start_Position'] == 12725998)].copy()

tcga_maf = tcga_maf.loc[~((tcga_maf['Tumor_Sample_Barcode'] == 'TCGA-FW-A3R5-06A-11D-A23B-08') &\
                    (tcga_maf['Chromosome'] == '1') & (tcga_maf['Start_Position'] == 12725998))]


def merge(tumor):
    tumor_df = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'] == tumor]
    tumor_df.sort_values(['Start_Position'], inplace=True)
    dfs = []
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
                        variant = result.iloc[[first]].copy()
                        variant['End_Position'] = result.iloc[last]['Start_Position']
                        variant['Variant_Classification'] = 'Missense_Mutation'
                        if last - first == 1:
                            type = 'DNP'
                        elif last - first == 2:
                            type = 'TNP'
                        else:
                            type = 'ONP'
                        variant['Variant_Type'] = type
                        ref = ''
                        alt = ''
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
                        ref_deviation = max([np.abs(ref_mean - ref) for ref in ref_counts])
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

tumors = tcga_maf['Tumor_Sample_Barcode'].unique()
data = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    for tumor, result in tqdm(zip(tumors, executor.map(merge, tumors))):
        data[tumor] = result

tcga_maf = pd.concat([data[i] for i in data if data[i] is not None] + [temp], ignore_index=True)
del data


def left_align(chr, start, seq):
    while True:
        if chromosomes[chr][start - len(seq): start] == seq:
            start = start - len(seq)
        else:
            break
    return start

def center_align(chr, start, seq, window, type):
    if type == 'insertion':
        repeats = len(re.findall('^(?:' + seq + ')*', chromosomes[chr][start: start + window * 2])[0]) / len(seq)
    else:
        repeats = len(re.findall('^(?:' + seq + ')*', chromosomes[chr][start + len(seq): start + len(seq) + window * 2])[0]) / len(seq)
    ## move start to middle of repeats, left preference
    shift = (repeats // 2) * len(seq)
    return int(shift)

def variant_features(maf, ref_length=20, alt_length=20, five_p_length=20, three_p_length=20):
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
        context_5p = np.nan
        context_3p = np.nan
        nan = False
        if pd.isna(Alt):
            nan = True
            print(str(index)+' Alt is nan')
        else:
            if not (len(re.findall('[AGCT]', Ref)) == 1 and len(re.findall('[AGCT]', Alt)) == 1):
                if Ref[-1] == Alt[-1] or Ref[0] == Alt[0]:
                    print('warning, variant not parsimonious', Ref, Alt)
                if len(re.findall('[AGCT]', Ref)) != len(re.findall('[AGCT]', Alt)):
                    if Ref == '-':
                        if chromosomes[Chr][Start - len(Alt): Start] == Alt:
                            Start = left_align(Chr, Start, Alt)
                        shift = center_align(Chr, Start, Alt, five_p_length + three_p_length, 'insertion')
                        Start = Start + shift
                        assert Start - five_p_length >= 0
                        context_5p = chromosomes[Chr][Start - five_p_length: Start]
                        context_3p = chromosomes[Chr][Start: Start + three_p_length]
                    elif Alt == '-':
                        Start = Start - 1
                        if chromosomes[Chr][Start - len(Ref): Start] == Ref:
                            shift = Start - left_align(Chr, Start, Ref)
                            Start = Start - shift
                            End = End - shift
                        shift = center_align(Chr, Start, Ref, five_p_length + three_p_length, 'deletion')
                        Start = Start + shift
                        End = End + shift
                        assert Start - five_p_length >= 0
                        context_5p = chromosomes[Chr][Start - five_p_length:Start]
                        context_3p = chromosomes[Chr][End:End + three_p_length]
                    else:
                        assert Start - (five_p_length + 1) >= 0
                        context_5p = chromosomes[Chr][Start - (five_p_length + 1):Start - 1]
                        context_3p = chromosomes[Chr][End:End + three_p_length]
                else:
                    assert Start - (five_p_length + 1) >= 0
                    context_5p = chromosomes[Chr][Start - (five_p_length + 1):Start - 1]
                    context_3p = chromosomes[Chr][End:End + three_p_length]

            else:
                assert Start-(five_p_length+1) >= 0
                context_5p = chromosomes[Chr][Start-(five_p_length+1):Start-1]
                context_3p = chromosomes[Chr][End:End+three_p_length]

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
        refs.append(Ref)
        alts.append(Alt)
        five_ps.append(context_5p)
        three_ps.append(context_3p)
        if not nan:
            if type(context_5p) != str:
                print(index, Ref, Alt, context_5p)
    return refs, alts, five_ps, three_ps

tcga_maf['Ref'], tcga_maf['Alt'], tcga_maf['five_p'], tcga_maf['three_p'] = variant_features(tcga_maf)

tcga_maf['index'] = tcga_maf.index.values

maf_pr = pr.PyRanges(tcga_maf.loc[:, ['Chromosome', 'Start_Position', 'End_Position', 'index']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))

##http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/simpleRepeat.txt.gz
repeats = pd.read_csv(file_path / 'simpleRepeat.txt', sep='\t', low_memory=False, header=None, usecols=[1, 2, 3])
repeats[1] = repeats[1].str.replace('chr', '')
repeats.rename(columns={1: 'Chromosome', 2: 'Start', 3: 'End'}, inplace=True)
repeats_pr = pr.PyRanges(repeats.loc[repeats['Chromosome'].isin(chromosomes)]).merge()

grs = {'repeat': repeats_pr}
result = pr.count_overlaps(grs, pr.concat({'maf': maf_pr}.values()))
result = result.df

tcga_maf = pd.merge(tcga_maf, result.iloc[:, 3:], how='left', on='index')

tcga_maf.drop(columns=['FILTER', 'NCALLERS', 'index'], inplace=True)

with open(file_path / 'tcga_controlled_maf.pkl', 'wb') as f:
    pickle.dump(tcga_maf, f)

tcga_sample_table = pd.read_csv(file_path / 'TCGA-CDR-SupplementalTableS1.tsv', sep='\t').iloc[:, 1:]
tcga_sample_table['histological_type'].fillna('', inplace=True)
tcga_sample_table = tcga_sample_table.loc[tcga_sample_table['bcr_patient_barcode'].isin(tcga_maf['Tumor_Sample_Barcode'].str[:12].unique())]

patient_to_sample = {i[:12]: i[:16] for i in tcga_maf['Tumor_Sample_Barcode'].unique()}
patient_to_barcode = {i[:12]: i for i in tcga_maf['Tumor_Sample_Barcode'].unique()}

tcga_sample_table['bcr_sample_barcode'] = tcga_sample_table['bcr_patient_barcode'].apply(lambda x: patient_to_sample[x])
tcga_sample_table['Tumor_Sample_Barcode'] = tcga_sample_table['bcr_patient_barcode'].apply(lambda x: patient_to_barcode[x])

ncit = pd.read_csv(file_path / 'NCIt_labels.tsv', sep='\t')
ncit.fillna('', inplace=True)
ncit_labels_dict = {i.type + '_' + i.histological_type: i.NCIt_label for i in ncit.itertuples()}
ncit_codes_dict = {i.type + '_' + i.histological_type: i.NCIt_code for i in ncit.itertuples()}
ncit_tmb_labels_dict = {i.type + '_' + i.histological_type: i.NCIt_tmb_label for i in ncit.itertuples()}
ncit_tmb_codes_dict = {i.type + '_' + i.histological_type: i.NCIt_tmb_code for i in ncit.itertuples()}

ncit_labels = []
ncit_codes = []
ncit_tmb_labels = []
ncit_tmb_codes = []

for row in tcga_sample_table.itertuples():
    ncit_labels.append(ncit_labels_dict[row.type + '_' + row.histological_type])
    ncit_codes.append(ncit_codes_dict[row.type + '_' + row.histological_type])
    ncit_tmb_labels.append(ncit_tmb_labels_dict[row.type + '_' + row.histological_type])
    ncit_tmb_codes.append(ncit_tmb_codes_dict[row.type + '_' + row.histological_type])

tcga_sample_table['NCIt_label'] = ncit_labels
tcga_sample_table['NCIt_code'] = ncit_codes
tcga_sample_table['NCIt_tmb_label'] = ncit_tmb_labels
tcga_sample_table['NCIt_tmb_code'] = ncit_tmb_codes

# add MANTIS data downloaded from 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5972025/bin/NIHMS962713-supplement-File_S2.xlsx'
mantis_file_name = 'NIHMS962713-supplement-File_S2.csv'
mantis_df = pd.read_csv(file_path / mantis_file_name, sep='\t')
mantis_df['Tumor_Sample_Barcode'] = mantis_df['Tumor Filename'].str.extract(r'(TCGA[^\.]+)\.')
tcga_sample_table = pd.merge(tcga_sample_table, mantis_df[['Tumor_Sample_Barcode', 'MANTIS Score']], how='left', on='Tumor_Sample_Barcode')

#add msi pcr labels, data processing and data sources available at ATGC/files/msi_ground_truth/
msi = pickle.load(open(file_path / 'msi_ground_truth' / 'msi_labels.pkl', 'rb'))
tcga_sample_table = pd.merge(tcga_sample_table, msi, how='left', left_on='bcr_patient_barcode', right_index=True)
tcga_sample_table.rename(columns={0: 'msi'}, inplace=True)

with open(file_path / 'tcga_controlled_sample_table.pkl', 'wb') as f:
    pickle.dump(tcga_sample_table, f)