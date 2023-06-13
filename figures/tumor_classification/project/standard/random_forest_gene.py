import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'data' / 'data.pkl', 'rb'))
tcga_maf = tcga_maf.loc[:, ['Tumor_Sample_Barcode', 'Hugo_Symbol']]
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
class_counts = dict(samples['type'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] > 125]
samples = samples.loc[samples['type'].isin(labels_to_use)]

gene_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "Hugo_Symbol"]).size().unstack(fill_value=0)
gene_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': gene_df.index, 'gene_counts': gene_df.values.tolist()})
samples = pd.merge(samples, gene_df, on='Tumor_Sample_Barcode', how='left')


del D, tcga_maf

A = samples['type'].astype('category')
classes = A.cat.categories.values
y_label = np.arange(len(classes))[A.cat.codes]

class_counts = dict(zip(*np.unique(y_label, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_label])
y_weights /= np.sum(y_weights)
gene_counts = np.stack(samples['gene_counts'].values)
gene_counts = np.log(gene_counts + 1)
predictions = []
test_idx = []
reg = RandomForestClassifier(n_estimators=900, min_samples_split=10, random_state=0, n_jobs=20)
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_label, y_label):
    print('fold')
    test_idx.append(idx_test)
    y_train, y_test = y_label[idx_train], y_label[idx_test]
    ##for context counts
    x_train, x_test = gene_counts[idx_train], gene_counts[idx_test]
    reg.fit(x_train, y_train,
            sample_weight=y_weights[idx_train] / np.sum(y_weights[idx_train])
            )
    predictions.append(reg.predict_proba(x_test))


with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'gene_forest.pkl', 'wb') as f:
    pickle.dump([predictions, y_label, test_idx], f)

##weighted accuracy
print(np.sum((np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], np.concatenate(predictions), multi_class='ovr'))

# 0.41793151209519097
# 0.4863891536913463
# 0.8982620207071613