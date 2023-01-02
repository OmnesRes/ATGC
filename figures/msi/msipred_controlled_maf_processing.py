import pandas as pd
import pickle
import pathlib
import MSIpred as mp

path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]

##path to files
path = cwd / 'files/'

D, maf, samples = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))
del D, samples

maf.drop(labels=['Matched_Norm_Sample_Barcode', 't_ref_count', 't_alt_count', 'Gene',
       'CDS_position', 'Ref', 'Alt', 'five_p', 'three_p', 'repeat',
       'index'], axis=1, inplace=True)

##MSIpred expects certain column names and chromosome format
maf.rename(columns={'STRAND': 'TRANSCRIPT_STRAND'})
maf['Chromosome'] = ['chr' + i for i in maf['Chromosome']]
maf.to_csv(path / 'msipred.maf', sep='\t')

##restrict repeats to 5 or less
simple_repeats = pd.read_csv(path / 'simpleRepeat.txt', sep='\t', low_memory=False, header=None)
simple_repeats = simple_repeats.loc[simple_repeats[5] <= 5]
simple_repeats.to_csv(path / 'simplerepeats_filtered.txt', sep='\t', header=False)

controlled_maf = mp.Raw_Maf(maf_path=str(path / 'msipred.maf'))
##this takes a long time
controlled_maf.create_tagged_maf(ref_repeats_file=str(path / 'simplerepeats_filtered.txt'), tagged_maf_file=str(path / 'tagged_controlled.maf'))
maf = pd.read_csv(path / 'tagged_controlled.maf', sep='\t', low_memory=False)

##get cancer labels
##selected tcga from the gdc data portal and downloaded the json for the 11,315 cases
import json
with open(path / 'cases.2020-02-28.json', 'r') as f:
    tcga_cancer_info = json.load(f)
cancer_labels = {i['submitter_id']: i['project']['project_id'].split('-')[-1] for i in tcga_cancer_info}

##MSIpred believes the exome kit for STAD was different than the other cancers
maf['Sample_ID'] = maf['Tumor_Sample_Barcode'].str[:12]
stad_maf = maf.loc[maf['Sample_ID'].apply(lambda x: cancer_labels[x[:12]]) == 'STAD']
other_maf = maf.loc[maf['Sample_ID'].apply(lambda x: cancer_labels[x[:12]]) != 'STAD']

stad_maf.to_csv(path / 'tagged_controlled_stad.maf', sep='\t')
other_maf.to_csv(path / 'tagged_controlled_other.maf', sep='\t')

stad_maf = mp.Tagged_Maf(tagged_maf_path=str(path / 'tagged_controlled_stad.maf'))
other_maf = mp.Tagged_Maf(tagged_maf_path=str(path / 'tagged_controlled_other.maf'))

stad_features = stad_maf.make_feature_table(exome_size=50)
other_features = other_maf.make_feature_table(exome_size=44)

all_features = other_features.append(stad_features)

with open(cwd / 'figures' / 'msi' / 'data' / 'msipred_features.pkl', 'wb') as f:
    pickle.dump(all_features, f)


