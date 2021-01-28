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
        elif pd.isna(col2):
            return col1
        else:
            return col1


with open(cwd / 'files' / 'msi_ground_truth' / 'TCGAbiolinks' / 'TCGAbiolinks_msi_labels.pkl', 'rb') as f:
    TCGAbiolinks = pickle.load(f)
TCGAbiolinks = pd.DataFrame(TCGAbiolinks)

with open(cwd / 'files' /  'msi_ground_truth' / 'cbioportal' / 'cbioportal_msi_labels.pkl', 'rb') as f:
    cbioportal = pickle.load(f)
cbioportal = pd.DataFrame(cbioportal)

with open(cwd / 'files' / 'msi_ground_truth' / 'papers' / 'paper_msi_labels.pkl', 'rb') as f:
    papers = pickle.load(f)
papers = pd.DataFrame(papers)


##merge biolinks with papers and give preference to papers (7 site assay)
papers_biolinks_joined = papers.join(TCGAbiolinks, how='outer', rsuffix='biolinks')
papers_biolinks_combined = papers_biolinks_joined.iloc[:, 0].combine(papers_biolinks_joined.iloc[:, 1], my_combine)
combined_df = pd.DataFrame(papers_biolinks_combined)
combined_df = combined_df.loc[combined_df.iloc[:, 0].isin(['MSS', 'MSI-H', 'MSI-L'])]

##cbioportal doesn't have any more cases

with open(cwd / 'files' / 'msi_ground_truth' / 'msi_labels.pkl', 'wb') as f:
    pickle.dump(combined_df, f)

