import pandas as pd
import numpy as np
import pickle
import pyranges as pr
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
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

##not every kit covered the entire exome, kit information is only available in the readgroups from the API
import requests
import json
cases = list(tcga_sample_table['bcr_patient_barcode'])

cases_endpt = 'https://api.gdc.cancer.gov/cases'
responses = []
step = 100
count = 0
X = True
while X:
    print(count)
    if (count+1)*step >= len(cases):
        value = cases[count*step:]
        X = False
    value = cases[count*step: (count+1)*step]
    count += 1
    filt = {"op": "in",
            "content": {
                "field": "cases.submitter_id",
                "value": value
            }
            }
    params = {'filters': json.dumps(filt), "expand": "files.analysis.metadata.read_groups", 'size': '100'}
    response = requests.get(cases_endpt, params=params).json()
    responses.append(response)

##If you don't want to use the API again save the data
with open(file_path / 'responses.pkl', 'wb') as f:
    pickle.dump([cases, responses], f)

with open(file_path / 'responses.pkl', 'rb') as f:
    cases, responses = pickle.load(f)


flattened_responses = []
for response in responses:
    for i in response["data"]["hits"]:
        flattened_responses.append(i)

##map a sample to its barcode
case_to_barcode = {i: j for i, j in zip(tcga_sample_table['bcr_patient_barcode'], tcga_sample_table['Tumor_Sample_Barcode'])}

##one case isn't in the API
case_to_barcode.pop('TCGA-AB-2852')

##will need this later
cancer_dict = {i: j for i, j in zip(tcga_sample_table['bcr_patient_barcode'], tcga_sample_table['type'])}

##the responses are not in order requested from the API
hits = {}
for case in case_to_barcode:
    for j in flattened_responses:
        for k in j['submitter_sample_ids']:
            if case in k:
                hits[case] = j
                break

data = {}
for case in case_to_barcode:
    for i in hits[case]['files']:
        for k in i['analysis']['metadata']['read_groups']:
            if k['library_strategy'] == 'WXS':
                Y = False
                sample = k['experiment_name']
                if sample[-2:] == '-1':
                    sample = sample[:-2]
                if sample == case_to_barcode[case]:
                    key = case_to_barcode[case]
                    Y = True
                else:
                    sample = sample[5:]
                    temp_sample = case_to_barcode[case][5:]
                    if sample == temp_sample:
                        key = case_to_barcode[case]
                        Y = True
                    else:
                        if cancer_dict[case] == 'LAML' and 'TCGA' not in k['experiment_name']:
                            if case in ['TCGA-AB-2918', 'TCGA-AB-2934', 'TCGA-AB-2864', 'TCGA-AB-2909', 'TCGA-AB-2807', 'TCGA-AB-2808', 'TCGA-AB-2935']:
                                pass
                            elif case == 'TCGA-AB-2869':
                                if k['experiment_name'] == "H_KA-141273-0927521":
                                    key = case_to_barcode[case]
                                    Y = True
                            else:
                                key = case_to_barcode[case]
                                Y = True
                        else:
                            if cancer_dict[case] == 'LUSC':
                                sample = k['experiment_name'][10:].replace('TP', '01A')
                                temp_sample = case_to_barcode[case][8:]
                                if sample == temp_sample:
                                    key = case_to_barcode[case]
                                    Y = True
                            else:
                                ##handle the mislabeled SKCM cases here
                                if case in ['TCGA-HR-A2OH', 'TCGA-HR-A2OG', 'TCGA-D9-A4Z6', 'TCGA-D9-A1X3']:
                                    sample = k['experiment_name'].replace('01A', '06A')
                                    if sample == case_to_barcode[case]:
                                        key = case_to_barcode[case]
                                        Y = True
                                else:
                                    # these BRCA cases have some sort of mislabel, not sure what to do:['TCGA-BH-A1EU', 'TCGA-A2-A04T', 'TCGA-E2-A15I']
                                    pass
                    if Y== False:
                        temp_sample = case_to_barcode[case][5:20]
                        if sample == temp_sample:
                            key = case_to_barcode[case]
                            Y = True
                if Y == True:
                    key = key[:15]
                    data[key] = data.get(key, {'centers': [], 'kits': [], 'beds': []})
                    data[key].update([('centers', data[key]['centers'] + [k['sequencing_center']]),\
                                         ('kits', data[key]['kits'] + [k['target_capture_kit_name']]),\
                                         ('beds', data[key]['beds'] + [k['target_capture_kit_target_region']])])



bad_kits=['Gapfiller_7m','NimbleGen Sequence Capture 2.1M Human Exome Array']
bad_beds=['https://bitbucket.org/cghub/cghub-capture-kit-info/raw/c38c4b9cb500b724de46546fd52f8d532fd9eba9/BI/vendor/Agilent/tcga_6k_genes.targetIntervals.bed',
'https://bitbucket.org/cghub/cghub-capture-kit-info/raw/c38c4b9cb500b724de46546fd52f8d532fd9eba9/BI/vendor/Agilent/cancer_2000gene_shift170.targetIntervals.bed']

bad_samples = []
null_samples = []

for sample in tcga_sample_table['Tumor_Sample_Barcode']:
    sample = sample[:15]
    if sample not in data:
        null_samples.append(sample)
    else:
        X = False
        for kit, bed in zip(data[sample]['kits'], data[sample]['beds']):
            if not kit:
                null_samples.append(sample)
            else:
                for sub_kit, sub_bed in zip(kit.split('|'), bed.split('|')):
                    if sub_kit not in bad_kits:
                        if sub_bed not in bad_beds:
                            X = True
                            break
        if X == False:
            bad_samples.append(sample)

##add columns to the sample table
tcga_sample_table['Exome_Covered'] = ~tcga_sample_table['Tumor_Sample_Barcode'].str[:15].isin(bad_samples + null_samples)
tcga_sample_table['Exome_Unknown'] = tcga_sample_table['Tumor_Sample_Barcode'].str[:15].isin(null_samples)

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

