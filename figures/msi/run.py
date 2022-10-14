import numpy as np
from model.Sample_MIL import RaggedModels, InstanceModels
from model.KerasLayers import Losses, Metrics
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from model import DatasetsUtils
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

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

weights = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_BE', min_delta=0.002, patience=20, mode='min', restore_best_weights=True)]
losses = [Losses.BinaryCrossEntropy(from_logits=True)]

##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=9, random_state=0, shuffle=True).split(y_strat, y_strat):
    while True:
        idx_train_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]
        ds_train = tf.data.Dataset.from_tensor_slices((idx_train_train))
        ds_train = ds_train.apply(DatasetsUtils.Apply.SubSample(batch_size=128, ds_size=len(idx_train_train)))

        ds_train = ds_train.map(lambda x: ((
            index_loader(x),
        )

        ),
                                )

        ds_train = ds_train.map(lambda x: ((five_p_loader(x[0], x[1]),
                                                three_p_loader(x[0], x[1]),
                                                ref_loader(x[0], x[1]),
                                                alt_loader(x[0], x[1]),
                                                strand_loader(x[0], x[1]),
                                            ),
                                           y_label_loader(x[0]),
                                           ))
        ds_train.prefetch(1)


        ds_valid = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(idx_valid),
                                               three_p_loader_eval(idx_valid),
                                               ref_loader_eval(idx_valid),
                                               alt_loader_eval(idx_valid),
                                               strand_loader_eval(idx_valid),
                                            ),
                                           tf.gather(y_label, idx_valid),
                                           ))
        ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
        sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=128)
        mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=y_label.shape[-1], output_types=['other'], mil_hidden=(256, 128), attention_layers=[], dropout=.5, instance_dropout=.5, regularization=.2, input_dropout=dropout)

        mil.model.compile(loss=losses,
                          metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        mil.model.fit(ds_train,
                      steps_per_epoch=10,
                      validation_data=ds_valid,
                      epochs=10000,
                      callbacks=callbacks,
                      # workers=4
                      )
        eval = mil.model.evaluate(ds_valid)
        if eval[1] < .07:
            break
        else:
            del mil
    weights.append(mil.model.get_weights())


with open(cwd / 'figures' / 'msi' / 'results' / 'run.pkl', 'wb') as f:
    pickle.dump(weights, f)




test_idx = []
predictions = []
for run, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=9, shuffle=True, random_state=0).split(y_strat, y_strat)):
    test_idx.append(idx_test)
    mil.model.set_weights(weights[run])
    ds_test = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(idx_test),
                                           three_p_loader_eval(idx_test),
                                           ref_loader_eval(idx_test),
                                           alt_loader_eval(idx_test),
                                           strand_loader_eval(idx_test),
                                        ),
                                       tf.gather(y_label, idx_test),
                                       ))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)

    mil.model.evaluate(ds_test)
    predictions.append(mil.model.predict(ds_test))

from sklearn.metrics import average_precision_score
scores = []
for prediction, idx_test in zip(predictions, test_idx):
    scores.append(average_precision_score(y_label[idx_test][:, 0], prediction[:, 0]))

#
# ds_test = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(idx_test),
#                                        three_p_loader_eval(idx_test),
#                                        ref_loader_eval(idx_test),
#                                        alt_loader_eval(idx_test),
#                                        strand_loader_eval(idx_test),
#                                     ),
#                                    tf.gather(y_label, idx_test),
#                                    ))
# ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
#
# mil.model.evaluate(ds_test)
#
# import pylab as plt
# import seaborn as sns
#
# attention = mil.attention_model.predict(ds_test).numpy()
# head_1 = np.concatenate([i[:, 0] for i in attention])
# # head_2 = np.concatenate([i[:, 1] for i in attention])
#
# test_indexes = np.concatenate(np.array([np.where(D['sample_idx'] == i)[0] for i in samples.index], dtype='object')[idx_test], axis=-1)
# labels_repeats = D['repeat'][test_indexes] == 1
#
# repeats = head_1[labels_repeats]
# non_repeats = head_1[~labels_repeats]
#
#
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# fig.subplots_adjust(top=0.975,
# bottom=0.07,
# left=0.11,
# right=0.98,
# hspace=0.14,
# wspace=0.04)
# sns.kdeplot(non_repeats, shade=True, gridsize=300, ax=ax1, alpha=1)
# sns.kdeplot(non_repeats, shade=False, gridsize=300, ax=ax1, alpha=1, color='k', linewidth=1)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['left'].set_visible(False)
# ax1.spines['bottom'].set_linewidth(1)
# ax1.set_xticks([])
# ax1.tick_params(axis='y', length=0, width=0, labelsize=8)
# ax1.set_ylabel('Variant Density (thousand)', fontsize=10)
# # ax1.set_xlim(.04, .2)
# ax1.set_title('Other', fontsize=12, loc='left', y=.87, x=.01)
# sns.kdeplot(repeats, shade=True, gridsize=300, ax=ax2, alpha=1)
# sns.kdeplot(repeats, shade=False, gridsize=300, ax=ax2, alpha=1, color='k', linewidth=1)
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.spines['bottom'].set_linewidth(1)
# ax2.set_xticks([])
# ax2.tick_params(axis='y', length=0, width=0, labelsize=8)
# ax2.set_xlabel('Attention', fontsize=12)
# ax2.set_ylabel('Variant Density (thousand)', fontsize=10, labelpad=8)
# # ax2.set_xlim(.04, .2)
# ax2.set_title('Simple Repeats', fontsize=12, loc='left', y=.87, x=.01)
# fig.canvas.draw()
# ax1.set_yticklabels([str(int(round(float(i.get_text())/100 * non_repeats.shape[0] / 1000, 0))) for i in ax1.get_yticklabels()])
# ax2.set_yticklabels([str(int(round(float(i.get_text())/100 * repeats.shape[0] / 1000, 0))) for i in ax2.get_yticklabels()])
# # plt.savefig(cwd / 'figures' / 'msi' / 'kde.pdf')
#
#
#
#
#
#
#
