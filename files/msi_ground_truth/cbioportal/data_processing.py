import pandas as pd
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]


def my_combine(col1, col2):
    if col1 == col2:
        return col1
    else:
        if pd.isna(col1):
            return col2
        else:
            return col1


##get the coadread msi data
coad_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'cbioportal' / 'coadread_tcga_pub_clinical_data.tsv', sep='\t', low_memory=False, usecols=['Patient ID', 'MSI Status'])
coad_msi.set_index('Patient ID', inplace=True)


##get the stad cancers data
stad_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'cbioportal' / 'stad_tcga_pub_clinical_data.tsv', sep='\t', low_memory=False, usecols=['Patient ID', 'MSI Status'])
stad_msi.set_index('Patient ID', inplace=True)

##outer join
coad_stad_joined = coad_msi.join(stad_msi, how='outer', lsuffix='coad')

##combine
combined = coad_stad_joined.iloc[:, 0].combine(coad_stad_joined.iloc[:, 1], my_combine)
combined.dropna(inplace=True)
combined_df = pd.DataFrame(combined)

##get the esca and more STAD? data

esca_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'cbioportal' / 'stes_tcga_pub_clinical_data.tsv', sep='\t', low_memory=False, usecols=['Patient ID', 'MSI Status'])
esca_msi.dropna(inplace=True)
esca_msi.set_index('Patient ID', inplace=True)

coad_stad_esca_joined = combined_df.join(esca_msi, how='outer', rsuffix='esca')
gastro_combined = coad_stad_esca_joined.iloc[:, 0].combine(coad_stad_esca_joined.iloc[:, 1], my_combine)
gastro_combined_df = pd.DataFrame(gastro_combined)

##get the UCEC data
ucec_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'cbioportal' / 'ucec_tcga_pub_clinical_data.tsv', sep='\t', low_memory=False, usecols=['Patient ID', 'MSI Status 7 Marker Call'])
ucec_msi.set_index('Patient ID', inplace=True)

all_joined = gastro_combined_df.join(ucec_msi, how='outer')
all_combined = all_joined.iloc[:,0].combine(all_joined.iloc[:, 1], my_combine)


with open(cwd / 'files' / 'msi_ground_truth' / 'cbioportal' / 'cbioportal_msi_labels.pkl', 'wb') as f:
    pickle.dump(all_combined, f)

