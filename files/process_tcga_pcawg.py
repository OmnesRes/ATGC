import pickle
import re
import numpy as np
import pandas as pd
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

file_path = cwd / 'files/'
#https://dcc.icgc.org/releases/PCAWG/data_releases/latest
pcawg_sample_table = pd.read_csv(file_path / 'pcawg_sample_sheet.tsv', sep='\t', low_memory=False)

pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['library_strategy'].str.contains('WGS')]
pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['donor_wgs_exclusion_white_gray'] == 'Whitelist']
pcawg_sample_table = pcawg_sample_table.loc[~pcawg_sample_table['dcc_specimen_type'].str.contains('Normal')]
pcawg_sample_table = pcawg_sample_table.loc[~pcawg_sample_table['dcc_specimen_type'].str.contains('Cell line')]

types = pcawg_sample_table.groupby('icgc_donor_id')['dcc_specimen_type'].apply(list).to_frame('types').reset_index()
samples = pcawg_sample_table.groupby('icgc_donor_id')['icgc_sample_id'].apply(list).to_frame('samples').reset_index()
result = pd.merge(types, samples, on='icgc_donor_id')
result = result.loc[result['types'].apply(lambda x: len(x) > 1)]

preference_order = ['Primary tumour - solid tissue',
                    'Primary tumour - other',
                    'Primary tumour - lymph node',
                    'Primary tumour - blood derived (peripheral blood)',
                    'Metastatic tumour - metastasis local to lymph node',
                    'Metastatic tumour - lymph node',
                    'Metastatic tumour - metastasis to distant location',
                    'Primary tumour - blood derived (bone marrow)']

samples_to_use = []
for row in result.itertuples():
    choice = None
    for preference in preference_order:
        for index, tumor_type in enumerate(row.types):
            if tumor_type == preference:
                choice = index
                break
        if choice is not None:
            continue
    samples_to_use.append(row.samples[choice])

chosen_samples = pcawg_sample_table.loc[pcawg_sample_table['icgc_sample_id'].isin(samples_to_use)]
pcawg_sample_table = pcawg_sample_table.loc[~pcawg_sample_table['icgc_donor_id'].isin(result.icgc_donor_id.unique())]
pcawg_sample_table = pd.concat([pcawg_sample_table, chosen_samples], ignore_index=True)

##map to histology
##https://dcc.icgc.org/releases/PCAWG/clinical_and_histology
pcawg_histology = pd.read_csv(file_path / 'pcawg_specimen_histology_August2016_v9.csv', sep=',', low_memory=False)

histology_mapping = {i: j for i, j in zip(pcawg_histology['icgc_sample_id'], pcawg_histology['histology_abbreviation'])}

pcawg_sample_table['histology'] = pcawg_sample_table['icgc_sample_id'].apply(lambda x: histology_mapping[x])

class_counts = dict(pcawg_sample_table['histology'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] >= 33]

pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['histology'].isin(labels_to_use)]
pcawg_sample_table.reset_index(drop=True, inplace=True)


usecols = ['Chromosome', 'Start_position', 'End_position', 'Reference_Allele', 'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode']
#https://icgc.bionimbus.org/files/0e8a845d-a4f4-40bc-890b-5472702d087c
tcga_maf = pd.read_csv('/home/janaya2/Desktop/dcc_download/final_consensus_passonly.snv_mnv_indel.tcga.controlled.maf', sep='\t',
                        usecols=usecols,
                        low_memory=False)


tcga_maf = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'].isin(pcawg_sample_table['aliquot_id'])]
tcga_maf.drop_duplicates(inplace=True)

sizes = dict(tcga_maf['Tumor_Sample_Barcode'].value_counts())
tumors_to_use = [i for i in sizes if sizes[i] < 200000]
tcga_maf = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'].isin(tumors_to_use)]

chromosomes = {}
for i in list(range(1, 23))+['X', 'Y']:
    with open(file_path / 'chromosomes' / ('chr' + str(i) + '.txt')) as f:
        chromosomes[str(i)] = f.read()

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
        Start = row.Start_position
        End = row.End_position
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

with open(file_path / 'tcga_pcawg_maf.pkl', 'wb') as f:
    pickle.dump(tcga_maf, f)
