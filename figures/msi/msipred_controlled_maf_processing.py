import pandas as pd
import pickle
import pathlib

path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

##path to files
path = cwd / 'files/'


mc3_maf = pd.read_csv(path / 'mc3.v0.2.8.CONTROLLED.maf', sep='\t', low_memory=False)

ground_truth_msi = pickle.load(open(path / 'msi_ground_truth' / 'msi_labels.pkl', 'rb')).to_dict()
mc3_maf['Sample_ID'] = mc3_maf['Tumor_Sample_Barcode'].str.extract(r'([^-]+-[^-]+-[^-]+-[\d]+)[^\d]')

##limit to samples that will be used to speed up processing
mc3_maf = mc3_maf.loc[mc3_maf['Sample_ID'].str[:-3].isin(ground_truth_msi[0]), :]

##same filters as before
filters = ['PASS', 'NonExonic,bitgt', 'NonExonic,bitgt,wga', 'NonExonic', 'NonExonic,wga', 'bitgt', 'bitgt,wga', 'wga', \
           'broad_PoN_v2', 'NonExonic,bitgt,broad_PoN_v2', 'NonExonic,bitgt,broad_PoN_v2,wga', 'NonExonic,broad_PoN_v2', \
           'broad_PoN_v2,wga', 'bitgt,broad_PoN_v2', 'NonExonic,broad_PoN_v2,wga', 'bitgt,broad_PoN_v2,wga', \
           'NonExonic,bitgt,native_wga_mix', 'NonExonic,native_wga_mix', 'bitgt,native_wga_mix', 'native_wga_mix', \
           'NonExonic,bitgt,broad_PoN_v2,native_wga_mix', 'broad_PoN_v2,native_wga_mix', 'NonExonic,broad_PoN_v2,native_wga_mix', \
           'bitgt,broad_PoN_v2,native_wga_mix']


mc3_maf = mc3_maf.loc[mc3_maf['FILTER'].isin(filters)]
mc3_maf = mc3_maf.loc[mc3_maf['Chromosome'] != 'MT']

# save to pickle
with open(path / 'controlled_maf_for_msipred.pkl', 'wb') as f:
    pickle.dump(mc3_maf, f)

with open(path / 'controlled_maf_for_msipred.pkl', 'rb') as f:
    maf = pickle.load(f)

##MSIpred expects certain column names and chromosome format
maf.rename(columns={'STRAND': 'TRANSCRIPT_STRAND'})
maf['Chromosome'] = ['chr' + i for i in maf['Chromosome']]
maf.to_csv(path / 'msipred.maf', sep='\t')

import MSIpred as mp
controlled_maf = mp.Raw_Maf(maf_path=str(path / 'msipred.maf'))
controlled_maf.create_tagged_maf(ref_repeats_file=str(path / 'simpleRepeat.txt'), tagged_maf_file=str(path / 'tagged_controlled.maf'))
##this takes a long time
tagged_controlled_maf = mp.Tagged_Maf(tagged_maf_path=str(path / 'tagged_controlled.maf'))

maf = pd.read_csv(path / 'tagged_controlled.maf', sep='\t', low_memory=False)

##get cancer labels
##selected tcga from the gdc data portal and downloaded the json for the 11,315 cases
import json
with open(path / 'cases.2020-02-28.json', 'r') as f:
    tcga_cancer_info = json.load(f)
cancer_labels = {i['submitter_id']: i['project']['project_id'].split('-')[-1] for i in tcga_cancer_info}

##MSIpred believes the exome kit for STAD was different than the other cancers

stad_maf = maf.loc[maf['Sample_ID'].apply(lambda x: cancer_labels[x[:12]]) == 'STAD']
other_maf = maf.loc[maf['Sample_ID'].apply(lambda x: cancer_labels[x[:12]]) != 'STAD']

stad_maf.to_csv(path / 'tagged_controlled_stad.maf', sep='\t')
other_maf.to_csv(path / 'tagged_controlled_other.maf', sep='\t')

stad_maf = mp.Tagged_Maf(tagged_maf_path=str(path / 'tagged_controlled_stad.maf'))
other_maf = mp.Tagged_Maf(tagged_maf_path=str(path / 'tagged_controlled_other.maf'))

stad_features = stad_maf.make_feature_table(exome_size=50)
other_features = other_maf.make_feature_table(exome_size=44)

all_features = other_features.append(stad_features)

with open(cwd / 'figures' / 'msi' / 'data' / 'msipred_features_new.pkl', 'wb') as f:
    pickle.dump(all_features, f)


