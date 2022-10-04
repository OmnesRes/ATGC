import numpy as np
import pickle
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve, classification_report
# import MSIpred as mp
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))


D, samples, sample_df = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))


msipred_features = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'msipred_features.pkl', 'rb'))
msipred_features.fillna(0, inplace=True)
msipred_features = pd.merge(sample_df[['Tumor_Sample_Barcode', 'msi_status']], msipred_features, how='left', left_on='Tumor_Sample_Barcode', right_index=True)
msipred_features['msi_status'] = msipred_features['msi_status'].apply(lambda x: 1 if x == 'high' else 0)

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

indexes = [np.where(D['sample_idx'] == idx) for idx in range(sample_df.shape[0])]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')


five_p_loader = DatasetsUtils.Map.FromNumpy(five_p, tf.int32)
three_p_loader = DatasetsUtils.Map.FromNumpy(three_p, tf.int32)
ref_loader = DatasetsUtils.Map.FromNumpy(ref, tf.int32)
alt_loader = DatasetsUtils.Map.FromNumpy(alt, tf.int32)
strand_loader = DatasetsUtils.Map.FromNumpy(strand, tf.float32)


# set y label and weights
y_label = samples['class']
cancer_labels = [i if i in ['STAD', 'UCEC', 'COAD'] else 'other' for i in samples['cancer']]
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(cancer_labels, y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(cancer_labels, y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)


with open(cwd / 'figures' / 'msi' / 'results' / 'run.pkl', 'rb') as f:
    weights = pickle.load(f)


predictions = []
evaluations = []
test_idx = []
msipred_predictions = []
all_latents = []

##stratified K fold for test
sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8])
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], output_dims=[2], pooling='sum', mil_hidden=(64, 64, 32, 16), output_types=['classification_probability'])
mil.model.compile(loss=[Losses.CrossEntropy(from_logits=False)],
                  metrics=[Metrics.CrossEntropy(from_logits=False), Metrics.Accuracy()],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                     clipvalue=10000))

for run, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=9, shuffle=True, random_state=0).split(y_strat, y_strat)):
    ##due to the y_strat levels not being constant this idx_train/idx_valid split is not deterministic
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]


    ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    ds_test = ds_test.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                           three_p_loader(x, ragged_output=True),
                                           ref_loader(x, ragged_output=True),
                                           alt_loader(x, ragged_output=True),
                                           strand_loader(x, ragged_output=True)
                                         ),
                                          y
                                          ))

    mil.model.set_weights(weights[run])
    predictions.append(mil.model.predict(ds_test))
    test_idx.append(idx_test)
    #
    evaluations.append(mil.model.evaluate(ds_test))
    idx_train_valid = np.concatenate([idx_train, idx_valid], axis=-1)
    train_features = msipred_features.iloc[idx_train_valid, 2:]
    test_features = msipred_features.iloc[idx_test, 2:]
    # new_model = mp.svm_training(training_X=train_features, training_y=list(msipred_features.iloc[idx_train_valid, 1]))
    # predicted_MSI = new_model.predict_proba(test_features)
    # msipred_predictions.append(predicted_MSI)

    latent = np.concatenate(mil.attention_model.predict(ds_test).to_list()).flat
    test_indexes = np.concatenate(np.array([np.where(D['sample_idx'] == i)[0] for i in range(y_label.shape[0])], dtype='object')[idx_test], axis=-1)
    labels_repeats = D['repeat'][test_indexes] == 1
    repeats = latent[labels_repeats]
    non_repeats = latent[~labels_repeats]
    all_latents.append([non_repeats, repeats])


with open(cwd / 'figures' / 'msi' / 'results' / 'latents.pkl', 'wb') as f:
    pickle.dump(all_latents, f)

with open(cwd / 'figures' / 'msi' / 'results' / 'for_mantis_plot.pkl', 'wb') as f:
    pickle.dump([predictions, test_idx, sample_df, y_label], f)


###metrics
##msipred requires MSI-H to be 1, but pandas made MSS 1 for us
y_true = y_label[:, 0][np.concatenate(test_idx)]
mil_pred = np.concatenate([np.argmin(i, axis=-1) for i in predictions])
# msipred_pred = np.concatenate([np.argmax(i, axis=-1) for i in msipred_predictions])
mantis_pred = sample_df['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)].apply(lambda x: 1 if x > .4 else 0).values
print(classification_report(y_true, mil_pred, digits=5))
print(classification_report(y_true, msipred_pred, digits=5))
print(classification_report(y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)], mantis_pred, digits=5))





##get the average prc curve

mil_recalls=[]
mil_precisions=[]
msipred_recalls=[]
msipred_precisions=[]
mil_scores=[]
msipred_scores=[]
for pred_mil, pred_msipred, idx_test in zip(predictions, msipred_predictions, test_idx):
    print('run')
    mil_run_recalls=[]
    mil_run_precisions=[]
    msipred_run_recalls=[]
    msipred_run_precisions=[]
    for i in np.concatenate([np.arange(0, .3, .0001), np.arange(.3, 1, .001)]):
        mil_run_recalls.append(recall_score(y_label[:, 0][idx_test], (pred_mil[:, 0] > i) * 1))
        mil_run_precisions.append(precision_score(y_label[:, 0][idx_test], (pred_mil[:, 0] > i) * 1))
        msipred_run_recalls.append(recall_score(y_label[:, 0][idx_test], (pred_msipred[:, 1] > i) * 1))
        msipred_run_precisions.append(precision_score(y_label[:, 0][idx_test], (pred_msipred[:, 1] > i) * 1))
    mil_recalls.append(mil_run_recalls)
    mil_precisions.append(mil_run_precisions)
    msipred_recalls.append(msipred_run_recalls)
    msipred_precisions.append(msipred_run_precisions)
    mil_scores.append(average_precision_score(y_label[:, 0][idx_test], pred_mil[:, 0]))
    msipred_scores.append(average_precision_score(y_label[:, 0][idx_test], pred_msipred[:, 1]))


with open(cwd / 'figures' / 'msi' / 'results' / 'for_prc_plot.pkl', 'wb') as f:
    pickle.dump([mil_recalls, mil_precisions, msipred_recalls, msipred_precisions, mil_scores, msipred_scores, sample_df, y_true, test_idx], f)