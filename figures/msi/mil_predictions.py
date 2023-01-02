import numpy as np
import pickle
import tensorflow as tf
from model.Sample_MIL import RaggedModels, InstanceModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.metrics import precision_score, recall_score, average_precision_score, classification_report
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))
del tcga_maf

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]
indexes = [np.where(D['sample_idx'] == idx) for idx in samples.index]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')
dropout = .4
index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=dropout)
five_p_loader = DatasetsUtils.Map.FromNumpyandIndices(five_p, tf.int16)
three_p_loader = DatasetsUtils.Map.FromNumpyandIndices(three_p, tf.int16)
ref_loader = DatasetsUtils.Map.FromNumpyandIndices(ref, tf.int16)
alt_loader = DatasetsUtils.Map.FromNumpyandIndices(alt, tf.int16)
strand_loader = DatasetsUtils.Map.FromNumpyandIndices(strand, tf.float32)

five_p_loader_eval = DatasetsUtils.Map.FromNumpy(five_p, tf.int16)
three_p_loader_eval = DatasetsUtils.Map.FromNumpy(three_p, tf.int16)
ref_loader_eval = DatasetsUtils.Map.FromNumpy(ref, tf.int16)
alt_loader_eval = DatasetsUtils.Map.FromNumpy(alt, tf.int16)
strand_loader_eval = DatasetsUtils.Map.FromNumpy(strand, tf.float32)

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

y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)

losses = [Losses.BinaryCrossEntropy(from_logits=True)]

with open(cwd / 'figures' / 'msi' / 'results' / 'run.pkl', 'rb') as f:
    test_idx, weights = pickle.load(f)

predictions = []
evaluations = []
all_latents = []

##stratified K fold for test
sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=128)
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=y_label.shape[-1], mil_hidden=(256, 128), attention_layers=[], dropout=.5, instance_dropout=.5, regularization=.2, input_dropout=dropout)
mil.model.compile(loss=losses,
                  metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

for weight, idx_test in zip(weights, test_idx):
    ds_test = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(idx_test),
                                                   three_p_loader_eval(idx_test),
                                                   ref_loader_eval(idx_test),
                                                   alt_loader_eval(idx_test),
                                                   strand_loader_eval(idx_test),
                                                  ),
                                                   tf.gather(y_label, idx_test),
                                                 ))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)

    mil.model.set_weights(weight)
    predictions.append(mil.model.predict(ds_test))
    attention = np.concatenate([i[:, 0] for i in mil.attention_model.predict(ds_test).numpy()])
    test_indexes = np.concatenate(np.array([np.where(D['sample_idx'] == i)[0] for i in samples.index], dtype='object')[idx_test], axis=-1)
    labels_repeats = D['repeat'][test_indexes] > 0
    repeats = attention[labels_repeats]
    non_repeats = attention[~labels_repeats]
    all_latents.append([non_repeats, repeats])

with open(cwd / 'figures' / 'msi' / 'results' / 'attention.pkl', 'wb') as f:
    pickle.dump(all_latents, f)


###metrics
y_true = y_label[:, 0][np.concatenate(test_idx)]
##pandas made MSI-H 0
y_true = 1 - y_true
mil_pred = np.concatenate([((1 - tf.nn.sigmoid(i).numpy()) > .5).astype(np.int32) for i in predictions])
mantis_pred = samples['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(samples['MANTIS Score'][np.concatenate(test_idx)].values)].apply(lambda x: 1 if x > .4 else 0).values
print(classification_report(y_true, mil_pred, digits=5))
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
        run_recalls.append(recall_score((1 - y_label[:, 0][idx_test]), ((1 - tf.nn.sigmoid(pred[:, 0]).numpy()) > i).astype(np.int32)))
        run_precisions.append(precision_score((1 - y_label[:, 0][idx_test]), ((1 - tf.nn.sigmoid(pred[:, 0]).numpy()) > i).astype(np.int32), zero_division=1))
    recalls.append(run_recalls)
    precisions.append(run_precisions)
    scores.append(average_precision_score((1 - y_label[:, 0][idx_test]), (1 - tf.nn.sigmoid(pred[:, 0]).numpy())))

with open(cwd / 'figures' / 'msi' / 'results' / 'mil_scores.pkl', 'wb') as f:
    pickle.dump([recalls, precisions, scores, predictions, samples, y_label, test_idx], f)

