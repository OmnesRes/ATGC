import os
os.environ['OPENBLAS_NUM_THREADS'] = '10'
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


D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'pcawg' / 'data' / 'data.pkl', 'rb'))
class_counts = dict(samples['histology'].value_counts())

context_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "contexts"]).size().unstack(fill_value=0)
context_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': context_df.index, 'context_counts': context_df.values.tolist()})
samples = pd.merge(samples, context_df, on='Tumor_Sample_Barcode', how='left')
del D, tcga_maf

A = samples['histology'].astype('category')
classes = A.cat.categories.values
y_label = np.arange(len(classes))[A.cat.codes]

class_counts = dict(zip(*np.unique(y_label, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_label])
y_weights /= np.sum(y_weights)
context_counts = np.apply_along_axis(lambda x: np.log(x + 1), 0, np.stack(samples['context_counts'].values))

context_test_predictions = []
test_idx = []
cancer_aucs = []

reg = LogisticRegression(max_iter=10000)
for idx_train, idx_test in StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(y_label, y_label):
    print('fold')
    test_idx.append(idx_test)
    y_train, y_test = y_label[idx_train], y_label[idx_test]
    ##for context counts
    context_train, context_test = context_counts[idx_train], context_counts[idx_test]
    reg.fit(context_train, y_train,
            sample_weight=y_weights[idx_train] / np.sum(y_weights[idx_train]) * len(idx_train)
            )
    context_test_predictions.append(reg.predict_proba(context_test))


predictions = context_test_predictions
with open(cwd / 'figures' / 'tumor_classification' / 'pcawg' / 'standard' / 'results' / 'context_logistic.pkl', 'wb') as f:
    pickle.dump([predictions, y_label, test_idx], f)
##weighted accuracy
print(np.sum((np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], np.concatenate(predictions), multi_class='ovr'))

# 0.7778767919787002
# 0.8121314237573716
# 0.9767590072103768