from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
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
tcga_maf = tcga_maf.loc[:, ['Tumor_Sample_Barcode', 'genome_position']]
samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
class_counts = dict(samples['type'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] > 125]
samples = samples.loc[samples['type'].isin(labels_to_use)]

# bin_size = 3095677412 // 3000

bin_size = 3095677412 // 100000

tcga_maf['bin'] = tcga_maf['genome_position'] // bin_size
bin_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "bin"]).size().unstack(fill_value=0)
bin_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': bin_df.index, 'bin_counts': bin_df.values.tolist()})
samples = pd.merge(samples, bin_df, on='Tumor_Sample_Barcode', how='left')

del D, tcga_maf

A = samples['type'].astype('category')
classes = A.cat.categories.values
y_label = np.arange(len(classes))[A.cat.codes]

class_counts = dict(zip(*np.unique(y_label, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_label])
y_weights /= np.sum(y_weights)
bin_counts = np.stack(samples['bin_counts'].values)
bin_counts = np.log(bin_counts + 1)

predictions = []
test_idx = []
reg = LogisticRegression(max_iter=10000)
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_label, y_label):
    print('fold')
    test_idx.append(idx_test)
    y_train, y_test = y_label[idx_train], y_label[idx_test]
    ##for context counts
    x_train, x_test = bin_counts[idx_train], bin_counts[idx_test]
    with threadpool_limits(limits=20, user_api='blas'):
        reg.fit(x_train, y_train,
                sample_weight=y_weights[idx_train] / np.sum(y_weights[idx_train]) * len(idx_train)
                )
    predictions.append(reg.predict_proba(x_test))


with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'bin_logistic_100k.pkl', 'wb') as f:
    pickle.dump([predictions, y_label, test_idx], f)
##weighted accuracy
print(np.sum((np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], np.concatenate(predictions), multi_class='ovr'))

##bin 3000
# 0.44297074005275433
# 0.4867069166401864
# 0.8927926851076243


##bin 100k
# 0.4972332195122757
# 0.5413621438406948
# 0.9260118736529926