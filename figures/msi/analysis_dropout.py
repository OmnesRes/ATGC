import numpy as np
import tensorflow as tf
from figures.msi.model.Sample_MIL import InstanceModels, RaggedModels
from figures.msi.model.KerasLayers import Losses, Metrics
from figures.msi.model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve, classification_report
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-3], True)
tf.config.experimental.set_visible_devices(physical_devices[-3], 'GPU')

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))


D, samples, sample_df = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]


indexes = [np.where(D['sample_idx'] == idx) for idx in range(sample_df.shape[0])]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')

dropout = .4
five_p_loader_eval = DatasetsUtils.Map.FromNumpy(five_p, tf.int32)
three_p_loader_eval = DatasetsUtils.Map.FromNumpy(three_p, tf.int32)
ref_loader_eval = DatasetsUtils.Map.FromNumpy(ref, tf.int32)
alt_loader_eval = DatasetsUtils.Map.FromNumpy(alt, tf.int32)
strand_loader_eval = DatasetsUtils.Map.FromNumpy(strand, tf.float32)


# set y label and weights
y_label = samples['class']
cancer_labels = [i if i in ['STAD', 'UCEC', 'COAD'] else 'other' for i in samples['cancer']]
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(cancer_labels, y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(cancer_labels, y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)


with open(cwd / 'figures' / 'msi' / 'results' / 'run_dropout.pkl', 'rb') as f:
    weights = pickle.load(f)


predictions = []
evaluations = []
test_idx = []
all_latents = []

##stratified K fold for test
sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=64)
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], output_dims=[2], pooling='sum', mil_hidden=(64, 64, 32, 16), output_types=['classification_probability'], input_dropout=dropout)

mil.model.compile(loss=[Losses.CrossEntropy(from_logits=False)],
                  metrics=[Metrics.CrossEntropy(from_logits=False), Metrics.Accuracy()],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                     clipvalue=10000))

for run, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=9, shuffle=True, random_state=0).split(y_strat, y_strat)):
    ##due to the y_strat levels not being constant this idx_train/idx_valid split is not deterministic

    ds_test = tf.data.Dataset.from_tensor_slices(((
                                                       five_p_loader_eval(idx_test),
                                                       three_p_loader_eval(idx_test),
                                                       ref_loader_eval(idx_test),
                                                       alt_loader_eval(idx_test),
                                                       strand_loader_eval(idx_test),
                                                   ),
                                                   (
                                                       y_label_loader(idx_test),
                                                   ),
    ))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    mil.model.set_weights(weights[run])
    predictions.append(mil.model.predict(ds_test))
    evaluations.append(mil.model.evaluate(ds_test))
    test_idx.append(idx_test)
    latent = np.concatenate(mil.attention_model.predict(ds_test).to_list()).flat
    test_indexes = np.concatenate(np.array([np.where(D['sample_idx'] == i)[0] for i in range(y_label.shape[0])], dtype='object')[idx_test], axis=-1)
    labels_repeats = D['repeat'][test_indexes] == 1
    repeats = latent[labels_repeats]
    non_repeats = latent[~labels_repeats]
    all_latents.append([non_repeats, repeats])



# with open(cwd / 'figures' / 'msi' / 'results' / 'latents_dropout_ones_strand.pkl', 'wb') as f:
#     pickle.dump(all_latents, f)


##get the average prc curve

mil_recalls=[]
mil_precisions=[]
mil_scores=[]
for pred_mil, idx_test in zip(predictions,  test_idx):
    print('run')
    mil_run_recalls=[]
    mil_run_precisions=[]
    # for i in np.concatenate([np.arange(0, .3, .0001), np.arange(.3, 1, .001)]):
    #     mil_run_recalls.append(recall_score(y_label[:, 0][idx_test], (pred_mil[:, 0] > i) * 1))
    #     mil_run_precisions.append(precision_score(y_label[:, 0][idx_test], (pred_mil[:, 0] > i) * 1))
    # mil_recalls.append(mil_run_recalls)
    # mil_precisions.append(mil_run_precisions)
    mil_scores.append(average_precision_score(y_label[:, 0][idx_test], pred_mil[:, 0]))

y_true = y_label[:, 0][np.concatenate(test_idx)]
average_precision = np.sum(np.array(mil_scores) * np.array([i.shape[0] for i in test_idx]) / len(y_true))
