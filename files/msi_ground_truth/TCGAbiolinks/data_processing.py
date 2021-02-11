import pandas as pd
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]

def my_combine(col1, col2):
    if col1 == col2:
        return col1
    else:
        if pd.isna(col1):
            return col2
        else:
            return col1


##get the coad msi data
coad_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'TCGAbiolinks' / 'msi_COAD.csv', sep=',', low_memory=False, usecols=['bcr_patient_barcode', 'mononucleotide_and_dinucleotide_marker_panel_analysis_status'])
coad_msi.set_index('bcr_patient_barcode', inplace=True)

##get the stad cancers data
stad_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'TCGAbiolinks' / 'msi_STAD.csv', sep=',', low_memory=False, usecols=['bcr_patient_barcode', 'mononucleotide_and_dinucleotide_marker_panel_analysis_status'])
stad_msi.set_index('bcr_patient_barcode', inplace=True)

##outer join
coad_stad_joined = coad_msi.join(stad_msi, how='outer', rsuffix='stad')
coad_stad_combined = coad_stad_joined.iloc[:, 0].combine(coad_stad_joined.iloc[:, 1], my_combine)

combined_df = pd.DataFrame(coad_stad_combined)

##get the read cancers data
read_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'TCGAbiolinks' / 'msi_READ.csv', sep=',', low_memory=False, usecols=['bcr_patient_barcode', 'mononucleotide_and_dinucleotide_marker_panel_analysis_status'])
read_msi.set_index('bcr_patient_barcode', inplace=True)

coad_stad_read_joined = combined_df.join(read_msi, how='outer', rsuffix='read')
coad_stad_read_combined = coad_stad_read_joined.iloc[:, 0].combine(coad_stad_read_joined.iloc[:, 1], my_combine)
combined_df = pd.DataFrame(coad_stad_read_combined)


##get the esca cancers data
esca_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'TCGAbiolinks' / 'msi_ESCA.csv', sep=',', low_memory=False, usecols=['bcr_patient_barcode', 'mononucleotide_and_dinucleotide_marker_panel_analysis_status'])
esca_msi.set_index('bcr_patient_barcode', inplace=True)

coad_stad_read_esca_joined = combined_df.join(esca_msi, how='outer', rsuffix='esca')
coad_stad_read_esca_combined = coad_stad_read_esca_joined.iloc[:, 0].combine(coad_stad_read_esca_joined.iloc[:, 1], my_combine)
combined_df = pd.DataFrame(coad_stad_read_esca_combined)


##get the ucec cancers data
ucec_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'TCGAbiolinks' / 'msi_UCEC.csv', sep=',', low_memory=False, usecols=['bcr_patient_barcode', 'mononucleotide_and_dinucleotide_marker_panel_analysis_status'])
ucec_msi.set_index('bcr_patient_barcode', inplace=True)

coad_stad_read_esca_ucec_joined = combined_df.join(ucec_msi, how='outer', rsuffix='ucec')
coad_stad_read_esca_ucec_combined = coad_stad_read_esca_ucec_joined.iloc[:, 0].combine(coad_stad_read_esca_ucec_joined.iloc[:, 1], my_combine)
combined_df = pd.DataFrame(coad_stad_read_esca_ucec_combined)


##get the ucs cancers data
ucs_msi = pd.read_csv(cwd / 'files' / 'msi_ground_truth' / 'TCGAbiolinks' / 'msi_UCS.csv', sep=',', low_memory=False, usecols=['bcr_patient_barcode', 'mononucleotide_and_dinucleotide_marker_panel_analysis_status'])
ucs_msi.set_index('bcr_patient_barcode', inplace=True)

all_joined = combined_df.join(ucs_msi, how='outer', rsuffix='ucs')
all_combined = all_joined.iloc[:, 0].combine(all_joined.iloc[:, 1], my_combine)


with open(cwd / 'files' / 'msi_ground_truth' / 'TCGAbiolinks' / 'TCGAbiolinks_msi_labels.pkl', 'wb') as f:
    pickle.dump(all_combined, f)

