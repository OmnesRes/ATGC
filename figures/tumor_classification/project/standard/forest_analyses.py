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


predictions, y_label, test_idx = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'context_forest.pkl', 'rb'))
class_counts = dict(zip(*np.unique(y_label, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_label])
y_weights /= np.sum(y_weights)
##weighted accuracy
print(np.sum((np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], np.concatenate(predictions), multi_class='ovr'))

# 0.48656468994436153
# 0.5010062493379939
# 0.9252687138345195

recalls = recall_score(y_label[np.concatenate(test_idx)], (np.argmax(np.concatenate(predictions), axis=-1)), average=None)
precisions = precision_score(y_label[np.concatenate(test_idx)], (np.argmax(np.concatenate(predictions), axis=-1)), average=None)

with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'context_forest_metrics.pkl', 'wb') as f:
    pickle.dump([precisions, recalls], f)

predictions, y_label, test_idx = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'gene_forest.pkl', 'rb'))
##weighted accuracy
print(np.sum((np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) * y_weights[np.concatenate(test_idx)]))
##unweighted accuracy
print(sum(np.argmax(np.concatenate(predictions), axis=-1) == y_label[np.concatenate(test_idx)]) / len(y_label))
print(roc_auc_score(y_label[np.concatenate(test_idx)], np.concatenate(predictions), multi_class='ovr'))

# 0.4165950169416204
# 0.4851181018959856
# 0.898216458794835

recalls = recall_score(y_label[np.concatenate(test_idx)], (np.argmax(np.concatenate(predictions), axis=-1)), average=None)
precisions = precision_score(y_label[np.concatenate(test_idx)], (np.argmax(np.concatenate(predictions), axis=-1)), average=None)

with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'standard' / 'results' / 'gene_forest_metrics.pkl', 'wb') as f:
    pickle.dump([precisions, recalls], f)