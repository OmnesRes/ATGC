import pandas as pd
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]

##get the coad msi data
coad_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'papers' / '2011-11-14592C-Sup Table 1.csv', sep='\t', low_memory=False, usecols=['patient', 'MSI_status'])
coad_msi.drop(coad_msi.tail(2).index, inplace=True)
coad_msi.set_index('patient', inplace=True)

##get the gastro cancers data
gastro_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'papers' / '1-s2.0-S1535610818301144-mmc2.csv', sep='\t', low_memory=False, skiprows=1, usecols=['TCGA Participant Barcode', 'MSI Status'])
gastro_msi.set_index('TCGA Participant Barcode', inplace=True)

##outer join
gastro_coad_joined = gastro_msi.join(coad_msi, how='outer')

def my_combine(col1, col2):
    if col1 == col2:
        return col1
    else:
        if pd.isna(col1):
            return col2
        elif pd.isna(col2):
            return col1
        else:
            return col1


##combine
combined = gastro_coad_joined.iloc[:,0].combine(gastro_coad_joined.iloc[:, 1], my_combine)
combined.dropna(inplace=True)
combined_df = pd.DataFrame(combined)

##get the UCEC data
ucec_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'papers' / 'datafile.S1.1.KeyClinicalData.csv', sep='\t', low_memory=False, usecols=['bcr_patient_barcode', 'msi_status_7_marker_call'])
ucec_msi.set_index('bcr_patient_barcode', inplace=True)

##another UCEC publication

other_ucec_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'papers' / '1-s2.0-S1535610818301193-mmc4.csv', sep='\t', low_memory=False, usecols=['Sample.ID', 'MSI_status'], skiprows=1)
other_ucec_msi.dropna(inplace=True)
other_ucec_msi.set_index('Sample.ID', inplace=True)

ucec_joined = ucec_msi.join(other_ucec_msi, how='outer')
ucec_combined = ucec_joined.iloc[:,0].combine(ucec_joined.iloc[:,1], my_combine)
ucec_combined_df = pd.DataFrame(ucec_combined)

all_joined = combined_df.join(ucec_combined_df, how='outer', rsuffix='ucec')
all_combined = all_joined.iloc[:,0].combine(all_joined.iloc[:, 1], my_combine)

with open(cwd / 'files' / 'msi_ground_truth' / 'papers' / 'paper_msi_labels.pkl', 'wb') as f:
    pickle.dump(all_combined, f)

