import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score, classification_report
import MSIpred as mp

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))
del D, tcga_maf

msipred_features = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'msipred_features.pkl', 'rb'))
msipred_features.fillna(0, inplace=True)
msipred_features = pd.merge(samples[['Tumor_Sample_Barcode', 'msi_status']], msipred_features, how='left', left_on='Tumor_Sample_Barcode', right_index=True)
msipred_features['msi_status'] = msipred_features['msi_status'].apply(lambda x: 1 if x == 'high' else 0)

A = samples.msi_status.astype('category')
classes = A.cat.categories.values

# set y label and weights
y_label = A.cat.codes.values[:, np.newaxis]
cancer_labels = [i if i in ['STAD', 'UCEC', 'COAD'] else 'other' for i in samples['type']]
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(cancer_labels, y_label[:, 0])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(cancer_labels, y_label[:, 0])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

predictions = []
evaluations = []
test_idx = []
all_latents = []
train_valids = []


for run, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=9, shuffle=True, random_state=0).split(y_strat, y_strat)):
    ##due to the y_strat levels not being constant this idx_train/idx_valid split is not deterministic
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]
    idx_train_valid = np.concatenate([idx_train, idx_valid], axis=-1)
    train_features = msipred_features.iloc[idx_train_valid, 2:]
    test_features = msipred_features.iloc[idx_test, 2:]
    new_model = mp.svm_training(training_X=train_features, training_y=list(msipred_features.iloc[idx_train_valid, 1]))
    predicted_MSI = new_model.predict_proba(test_features)
    predictions.append(predicted_MSI)
    test_idx.append(idx_test)
    train_valids.append(idx_train_valid)




###metrics
y_true = y_label[:, 0][np.concatenate(test_idx)]
##pandas made MSI-H 0
y_true = 1 - y_true
msipred_pred = np.concatenate([np.argmax(i, axis=-1) for i in predictions])
mantis_pred = samples['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)].apply(lambda x: 1 if x > .4 else 0).values
print(classification_report(y_true, msipred_pred, digits=5))
print(classification_report(y_true[~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)], mantis_pred, digits=5))


##get the average prc curve
recalls = []
precisions = []
scores = []
for pred, idx_test in zip(predictions, test_idx):
    print('run')
    run_recalls = []
    run_precisions = []
    for i in np.concatenate([np.arange(0, .001, .000001), np.arange(.001, .1, .0001), np.arange(.1, .9, .001), np.arange(.9, .999, .0001), np.arange(.999, 1, .000001)]):
        run_recalls.append(recall_score(1 - y_label[:, 0][idx_test], (pred[:, 1] > i).astype(np.int32)))
        run_precisions.append(precision_score(1 - y_label[:, 0][idx_test], (pred[:, 1] > i).astype(np.int32), zero_division=1))
    recalls.append(run_recalls)
    precisions.append(run_precisions)
    scores.append(average_precision_score((1 - y_label[:, 0][idx_test]), pred[:, 1]))



with open(cwd / 'figures' / 'msi' / 'results' / 'msipred_scores.pkl', 'wb') as f:
    pickle.dump([recalls, precisions, scores, predictions, samples, y_label, test_idx, train_valids], f)

