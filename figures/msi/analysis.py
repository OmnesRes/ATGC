import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve, classification_report
import pylab as plt
import MSIpred as mp

disable_eager_execution()
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

from figures.msi.model.MSIModel import InputFeatures, ATGC
from model.CustomKerasTools import BatchGenerator, Losses


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

features = [InputFeatures.VariantSequence(20, 4, 2, [8, 8, 8, 8],
                                         {'5p': D['seq_5p'], '3p': D['seq_3p'], 'ref': D['seq_ref'], 'alt': D['seq_alt'], 'strand': D['strand_emb'], 'cds': D['cds_emb']},
                                         use_frame=False)]

sample_features = ()
# set y label and weights
y_label = samples['class']
cancer_labels = [i if i in ['STAD', 'UCEC', 'COAD'] else 'other' for i in samples['cancer']]
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(cancer_labels, y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(cancer_labels, y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

atgc = ATGC(features, latent_dimension=64)
atgc.build_instance_encoder_model(return_latent=False)
atgc.build_mil_model(output_dim=y_label.shape[1], output_extra=1, output_type='classification_probability', aggregation='recursion', mil_hidden=(32, 16))
metrics = [Losses.Weighted.CrossEntropy.cross_entropy, Losses.Weighted.Accuracy.accuracy]
atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=10000), loss=Losses.Weighted.CrossEntropy.cross_entropy, metrics=metrics)


with open(cwd / 'figures' / 'msi' / 'results' / 'run.pkl', 'rb') as f:
    weights = pickle.load(f)

predictions = []
test_idx = []
msipred_predictions = []
##stratified K fold for test
for run, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=9, random_state=0).split(y_strat, y_strat)):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    data_test = next(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                    y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx_test).data_generator())

    atgc.mil_model.set_weights(weights[run])
    predictions.append(atgc.mil_model.predict(data_test[0]))
    test_idx.append(idx_test)

    idx_train_valid = np.concatenate([idx_train, idx_valid], axis=-1)
    train_features = msipred_features.iloc[idx_train_valid, 2:]
    test_features = msipred_features.iloc[idx_test, 2:]
    new_model = mp.svm_training(training_X=train_features, training_y=list(msipred_features.iloc[idx_train_valid, 1]))
    predicted_MSI = new_model.predict_proba(test_features)
    msipred_predictions.append(predicted_MSI)


###metrics
##msipred requires MSI-H to be 1, but pandas made MSS 1 for us
y_true = y_label[:, 0][np.concatenate(test_idx)]
mil_pred = np.concatenate([np.argmin(i[0, :, :-1], axis=-1) for i in predictions])
msipred_pred = np.concatenate([np.argmax(i, axis=-1) for i in msipred_predictions])
mantis_pred = sample_df['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)].apply(lambda x: 1 if x > .4 else 0).values
print(classification_report(y_true, mil_pred, digits=5))
print(classification_report(y_true, msipred_pred, digits=5))
print(classification_report(y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)], mantis_pred, digits=5))


##compare atgc, mantis predictions

fig = plt.figure(figsize=(22.62372, 12))
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.99)
fig.subplots_adjust(left=.08)
fig.subplots_adjust(right=.99)
ax.scatter(np.concatenate([i[0, :, 0] for i in predictions])[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)][y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 0],
            sample_df['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)].values[y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 0],
           s=sample_df['all_counts'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)][y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 0].values/1000,
           edgecolor='k', linewidths=.1, label='MSI Low')

scatter = ax.scatter(np.concatenate([i[0, :, 0] for i in predictions])[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)][y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 1],
            sample_df['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)].values[y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 1],
           s=sample_df['all_counts'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)][y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)] == 1].values/1000,
           edgecolor='k', linewidths=.1, label='MSI High')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['left'].set_position(['outward', -5])
ax.spines['left'].set_bounds(.2, 1.4)
ax.spines['bottom'].set_linewidth(1)
ax.spines['bottom'].set_bounds(0, 1)
ax.tick_params(axis='x', length=4, width=1, labelsize=8)
ax.tick_params(axis='y', length=4, width=1, labelsize=8)
ax.set_xlabel('ATGC')
ax.set_ylabel('MANTIS')
legend1 = ax.legend(frameon=False, loc=(.15,.88), ncol=2, columnspacing=.2, handletextpad=0)
legend1.legendHandles[0].set_sizes([30])
legend1.legendHandles[1].set_sizes([30])
ax.add_artist(legend1)
handles, labels = scatter.legend_elements(prop='sizes', num=5, alpha=1, color='w', markeredgecolor='k', markeredgewidth=1)
ax.legend(handles, labels, frameon=False, title='TMB', loc=(.03, .59))
plt.savefig(cwd / 'figures' / 'msi' / 'atgc_mantis.pdf')


##plot the average prc curve

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
        mil_run_recalls.append(recall_score(y_label[:, 0][idx_test], (pred_mil[0, :, 0] > i) * 1))
        mil_run_precisions.append(precision_score(y_label[:, 0][idx_test], (pred_mil[0, :, 0] > i) * 1))
        msipred_run_recalls.append(recall_score(y_label[:, 0][idx_test], (pred_msipred[:, 1] > i) * 1))
        msipred_run_precisions.append(precision_score(y_label[:, 0][idx_test], (pred_msipred[:, 1] > i) * 1))
    mil_recalls.append(mil_run_recalls)
    mil_precisions.append(mil_run_precisions)
    msipred_recalls.append(msipred_run_recalls)
    msipred_precisions.append(msipred_run_precisions)
    mil_scores.append(average_precision_score(y_label[:, 0][idx_test], pred_mil[0, :, 0]))
    msipred_scores.append(average_precision_score(y_label[:, 0][idx_test], pred_msipred[:, 1]))

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.1)
fig.subplots_adjust(top=.98)
fig.subplots_adjust(left=.1)
fig.subplots_adjust(right=.98)
average_precision = np.sum(np.array(mil_scores) * np.array([i.shape[0] for i in test_idx]) / len(y_true))
ax.plot(np.mean(mil_recalls, axis=0), np.mean(mil_precisions,axis=0), linewidth=1, label=f"ATGC:{f'{average_precision:.3f}':>10}")
average_precision = np.sum(np.array(msipred_scores) * np.array([i.shape[0] for i in test_idx]) / len(y_true))
ax.plot(np.mean(msipred_recalls, axis=0), np.mean(msipred_precisions,axis=0), linewidth=1, label=f"MSIpred:{f'{average_precision:.3f}':>6}")
precision, recall, _ = precision_recall_curve(y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)], sample_df['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)].values)
average_precision = average_precision_score(y_true[~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)], sample_df['MANTIS Score'][np.concatenate(test_idx)][~np.isnan(sample_df['MANTIS Score'][np.concatenate(test_idx)].values)].values)
ax.plot(recall, precision, linewidth=1, color='red', label=f"MANTIS:{f'{average_precision:.4f}':>7}")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['left'].set_bounds(.2, 1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['bottom'].set_bounds(0, 1)
ax.tick_params(axis='x', length=5, width=1, labelsize=8)
ax.tick_params(axis='y', length=5, width=1, labelsize=8)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.legend(fontsize=8, borderpad=0, title='Average Precision Score', title_fontsize=10, frameon=False, loc=(.04, .78))
plt.savefig(cwd / 'figures' / 'msi' / 'prc.pdf')


all_latents = []
##save latents for graphing
for index, i in enumerate(weights):
    atgc.mil_model.set_weights(i)
    latent = atgc.intermediate_model.predict(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                    y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=test_idx[index]).data_generator(), steps=1)

    test_indexes = np.concatenate(np.array([np.where(D['sample_idx'] == i)[0] for i in range(y_label.shape[0])])[test_idx[index]], axis=-1)
    labels_repeats = D['repeat'][test_indexes] == 1
    repeats = latent[labels_repeats]
    non_repeats = latent[~labels_repeats]
    all_latents.append([non_repeats, repeats])

with open(cwd / 'figures' / 'msi' / 'results' / 'latents.pkl', 'wb') as f:
    pickle.dump(all_latents, f)





