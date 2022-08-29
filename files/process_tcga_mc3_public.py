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

usecols = ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'STRAND', 'Variant_Classification', 'Variant_Type', 'Gene', 'Reference_Allele', 'Tumor_Seq_Allele2', 't_ref_count', 't_alt_count', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode', 'FILTER', 'CDS_position']

# tcga mc3 filename download from 'https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc'
tcga_maf = pd.read_csv(file_path / 'mc3.v0.2.8.PUBLIC.maf', sep='\t', usecols=usecols, low_memory=False)

##The MAF contains nonpreferred pairs which results in some samples having duplicated variants
tcga_maf = tcga_maf.loc[(tcga_maf['FILTER'] == 'PASS') | (tcga_maf['FILTER'] == 'wga') | (tcga_maf['FILTER'] == 'native_wga_mix')]
tcga_maf = tcga_maf.loc[~pd.isna(tcga_maf['Tumor_Seq_Allele2'])]

tumor_to_normal = {}

grouped = tcga_maf[['Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode']].groupby(['Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode']).size().reset_index()

for i in grouped.itertuples():
    tumor_to_normal[i.Tumor_Sample_Barcode] = tumor_to_normal.get(i.Tumor_Sample_Barcode, []) + [i.Matched_Norm_Sample_Barcode]

for i in tumor_to_normal:
    tumor_to_normal[i] = set(tumor_to_normal[i])

##gdc data portal metadata files for TCGA WXS bams.  multiple files because only 10k can be added to the cart at a time.
with open(file_path / 'bams' / 'first_part.json', 'r') as f:
    metadata = json.load(f)

with open(file_path / 'bams' / 'second_part.json', 'r') as f:
    metadata += json.load(f)

with open(file_path / 'bams' / 'third_part.json', 'r') as f:
    metadata += json.load(f)

sample_to_id = {}
for i in metadata:
    sample_to_id[i['associated_entities'][0]['entity_submitter_id']] = i['associated_entities'][0]['entity_id']

cmd = ['ls', 'files/beds']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
files = [str(i, 'utf-8') for i in p.communicate()[0].split() if '.bed' in str(i)[-5:]]


tumor_to_bed = {}
for i in tumor_to_normal:
    if i in sample_to_id and list(tumor_to_normal[i])[0] in sample_to_id:
        for j in files:
            if sample_to_id[i] in j and sample_to_id[list(tumor_to_normal[i])[0]] in j:
                tumor_to_bed[i] = j

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

def get_overlap(tumor):
    file = tumor_to_bed[tumor]
    try:
        bed_df = pd.read_csv(file_path / 'beds' / file, names=['Chromosome', 'Start', 'End'], low_memory=False, sep='\t')
    except:
        return None
    bed_df = bed_df.loc[bed_df['Chromosome'].isin(chromosomes)]
    bed_pr = pr.PyRanges(bed_df).merge()
    bed_size = sum([i + 1 for i in bed_pr.lengths()])
    tumor_df = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'] == tumor]
    tumor_df['index'] = tumor_df.index.values
    tumor_pr = pr.PyRanges(tumor_df[['Chromosome', 'Start_Position', 'End_Position', 'index']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))
    grs = {'bed': bed_pr}
    result = pr.count_overlaps(grs, pr.concat({'maf': tumor_pr}.values()))
    result = result.df
    tumor_df = pd.merge(tumor_df, result.iloc[:, 3:], how='left', on='index')
    tumor_df = tumor_df.loc[tumor_df['bed'] > 0]
    if len(tumor_df) == 0:
        return None
    tumor_df.drop(columns=['bed', 'index'], inplace=True)
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
    return tumor_df, bed_size

data = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for tumor, result in tqdm(zip(tumor_to_bed.keys(), executor.map(get_overlap, tumor_to_bed.keys()))):
        data[tumor] = result

tcga_maf = pd.concat([data[i][0] for i in data if data[i] is not None] + [temp], ignore_index=True)

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

##chang hotspots https://github.com/taylor-lab/hotspots/blob/master/publication_hotspots.vcf
hotspots = pd.read_csv(cwd / 'files' / 'hotspots.vcf', skiprows=1, sep='\t')
hotspots = hotspots[['#CHROM', 'POS', 'REF', 'ALT']].rename(columns={'#CHROM': 'Chromosome', 'POS': 'Start_Position', 'REF': 'Reference_Allele', 'ALT': 'Tumor_Seq_Allele2'})
hotspots.drop_duplicates(inplace=True)
hotspots['hotspot'] = True
tcga_maf = pd.merge(tcga_maf, hotspots, how='left', on=['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])

with open(file_path / 'tcga_public_maf.pkl', 'wb') as f:
    pickle.dump(tcga_maf, f)

tcga_sample_table = pd.read_csv(file_path / 'TCGA-CDR-SupplementalTableS1.tsv', sep='\t').iloc[:, 1:]
tcga_sample_table['histological_type'].fillna('', inplace=True)
tcga_sample_table = tcga_sample_table.loc[tcga_sample_table['bcr_patient_barcode'].isin(tcga_maf['Tumor_Sample_Barcode'].str[:12].unique())]
patient_to_sample = {i[:12]: i[:16] for i in tcga_maf['Tumor_Sample_Barcode'].unique()}
patient_to_barcode = {i[:12]: i for i in tcga_maf['Tumor_Sample_Barcode'].unique()}

tcga_sample_table['bcr_sample_barcode'] = tcga_sample_table['bcr_patient_barcode'].apply(lambda x: patient_to_sample[x])
tcga_sample_table['Tumor_Sample_Barcode'] = tcga_sample_table['bcr_patient_barcode'].apply(lambda x: patient_to_barcode[x])
tcga_sample_table['coverage'] = tcga_sample_table['Tumor_Sample_Barcode'].apply(lambda x: data[x][1])


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

with open(file_path / 'tcga_public_sample_table.pkl', 'wb') as f:
    pickle.dump(tcga_sample_table, f)
