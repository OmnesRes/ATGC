import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

predictions, y_label, test_idx = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'mil_encoder' / 'results' / 'gene_predictions.pkl', 'rb'))

test_idx, predictions = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'gene_predictions.pkl', 'rb'))
y_label = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_label, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_label])
y_weights /= np.sum(y_weights)

##weighted accuracy
print(np.sum((np.argmax(predictions, axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(predictions, axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], predictions, multi_class='ovr'))

# 0.5257284609464159
# 0.5555555555555556
# 0.92897640353388

recalls = recall_score(y_label[np.concatenate(test_idx)], np.argmax(predictions, axis=-1), average=None)
precisions = precision_score(y_label[np.concatenate(test_idx)], np.argmax(predictions, axis=-1), average=None)

with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'gene_metrics.pkl', 'wb') as f:
    pickle.dump([precisions, recalls], f)

test_idx, predictions = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'context_predictions.pkl', 'rb'))
##weighted accuracy
print(np.sum((np.argmax(predictions, axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(predictions, axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], predictions, multi_class='ovr'))

# 0.5269553129679649
# 0.5078911132295307
# 0.9357473407903015

recalls = recall_score(y_label[np.concatenate(test_idx)], np.argmax(predictions, axis=-1), average=None)
precisions = precision_score(y_label[np.concatenate(test_idx)], np.argmax(predictions, axis=-1), average=None)

with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'context_metrics.pkl', 'wb') as f:
    pickle.dump([precisions, recalls], f)