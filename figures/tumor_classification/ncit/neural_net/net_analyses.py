import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

predictions, y_label, test_idx = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'neural_net' / 'results' / 'gene_predictions.pkl', 'rb'))
y_label = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_label, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_label])
y_weights /= np.sum(y_weights)

##weighted accuracy
print(np.sum((np.argmax(predictions, axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(predictions, axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], predictions, multi_class='ovr'))

# 0.5088390462079297
# 0.5427609427609428
# 0.9279917230121018

recalls = recall_score(y_label[np.concatenate(test_idx)], np.argmax(predictions, axis=-1), average=None)
precisions = precision_score(y_label[np.concatenate(test_idx)], np.argmax(predictions, axis=-1), average=None)

with open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'neural_net' / 'results' / 'gene_metrics.pkl', 'wb') as f:
    pickle.dump([precisions, recalls], f)

test_idx, predictions = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'neural_net' / 'results' / 'context_predictions.pkl', 'rb'))
##weighted accuracy
print(np.sum((np.argmax(predictions, axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(predictions, axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], predictions, multi_class='ovr'))

# 0.45050134439031453
# 0.4696969696969697
# 0.9266788948167651

recalls = recall_score(y_label[np.concatenate(test_idx)], np.argmax(predictions, axis=-1), average=None)
precisions = precision_score(y_label[np.concatenate(test_idx)], np.argmax(predictions, axis=-1), average=None)

with open(cwd / 'figures' / 'tumor_classification' / 'ncit' / 'neural_net' / 'results' / 'context_metrics.pkl', 'wb') as f:
    pickle.dump([precisions, recalls], f)