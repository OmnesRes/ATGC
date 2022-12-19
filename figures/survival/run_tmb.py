import numpy as np
import pandas as pd
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from lifelines.utils import concordance_index
from model import DatasetsUtils
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-2], True)
tf.config.experimental.set_visible_devices(physical_devices[-2], 'GPU')

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'data' / 'data.pkl', 'rb'))
tcga_maf = tcga_maf.loc[:, ['Tumor_Sample_Barcode', 'contexts']]

context_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "contexts"]).size().unstack(fill_value=0)
context_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': context_df.index, 'context_counts': context_df.values.tolist()})

labels_to_use = ['BLCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC']
samples = samples.loc[samples['type'].isin(labels_to_use)]
samples.dropna(axis=0, subset=['OS', 'OS.time'], inplace=True)
samples = pd.merge(samples, context_df, on='Tumor_Sample_Barcode', how='left')

A = samples['type'].astype('category')
types_onehot = np.eye(len(labels_to_use))[A.cat.codes]
cancer_strat = np.argmax(types_onehot, axis=-1)
y_label = np.stack(np.concatenate([samples['OS.time'].values[:, np.newaxis], samples['OS'].values[:, np.newaxis], cancer_strat[:, np.newaxis]], axis=-1))

strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(samples['type'], y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(samples['type'], y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))

context_counts = np.sum(np.stack(samples['context_counts'].values), axis=-1, keepdims=True)
# context_counts = np.stack(samples['age_at_initial_pathologic_diagnosis'].values)[:, np.newaxis]
context_counts = np.log(context_counts)
# context_counts = np.random.permutation(context_counts)
context_counts = np.concatenate([context_counts, types_onehot], axis=-1)

contexts_loader = DatasetsUtils.Map.FromNumpy(context_counts, tf.float32)
y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)

predictions = []
test_evaluations = []
all_evaluations = []
test_idx = []
weights = []

cancer_test_ranks = {}
cancer_test_indexes = {}

ds_all = tf.data.Dataset.from_tensor_slices((
    (
        context_counts,
    ),
    (
        y_label,
    ),
))

ds_all = ds_all.batch(len(y_label), drop_remainder=False)

losses = [Losses.CoxPH(cancers=len(labels_to_use))]
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_coxph', min_delta=0.0001, patience=50, mode='min', restore_best_weights=True)]
context_encoder = InstanceModels.Feature(shape=(context_counts.shape[-1]), input_dropout=0, layers=[], regularization=0, layer_dropouts=[])
mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[context_encoder.model], output_dims=[1], mil_hidden=[128, 64, 32], mode='none', dropout=0)
mil.model.compile(loss=losses,
                  metrics=[Losses.CoxPH(cancers=len(labels_to_use))],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                     ))
initial_weights = mil.model.get_weights()

for idx_train, idx_test in StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(y_strat, y_strat):
    test_idx.append(idx_test)
    eval = 100
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=len(idx_train) // 4, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=len(idx_train), ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x: ((
                                          contexts_loader(x),
                                           ),
                                          (
                                          y_label_loader(x),
                                          ),
                                          )
                            )

    ds_valid = tf.data.Dataset.from_tensor_slices((
                                                  (
                                                   context_counts[idx_valid],
                                                   ),
                                                  (
                                                   y_label[idx_valid],
                                                  ),
                                                   ))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)

    ds_test = tf.data.Dataset.from_tensor_slices((
                                                 (
                                                  context_counts[idx_test],
                                                 ),
                                                 (
                                                  y_label[idx_test],
                                                 ),
                                                  ))

    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    for run in range(3):
        mil.model.set_weights(initial_weights)
        mil.model.fit(ds_train,
                      steps_per_epoch=2,
                      epochs=20000,
                      validation_data=ds_valid,
                      callbacks=callbacks)
        run_eval = mil.model.evaluate(ds_valid)[0]
        if run_eval < eval:
            eval = run_eval
            run_weights = mil.model.get_weights()
            print('test_eval', mil.model.evaluate(ds_test))
    mil.model.set_weights(run_weights)
    y_pred_all = mil.model.predict(ds_all)
    ##get ranks per cancer
    for index, cancer in enumerate(labels_to_use):
        mask = np.where(cancer_strat == index)[0]
        cancer_test_indexes[cancer] = cancer_test_indexes.get(cancer, []) + [mask[np.isin(mask, idx_test, assume_unique=True)]]
        temp = np.exp(-y_pred_all[mask, 0]).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(mask))
        cancer_test_ranks[cancer] = cancer_test_ranks.get(cancer, []) + [ranks[np.isin(mask, idx_test, assume_unique=True)]]

    weights.append(run_weights)
    all_evaluations.append(mil.model.evaluate(ds_all)[0])
    test_evaluations.append(mil.model.evaluate(ds_test)[0])

concordances = {}
for cancer in labels_to_use:
    indexes = np.concatenate(cancer_test_indexes[cancer])
    ranks = np.concatenate(cancer_test_ranks[cancer])
    print(cancer, concordance_index(samples['OS.time'][indexes], ranks, samples['OS'][indexes]))
    concordances[cancer] = concordance_index(samples['OS.time'][indexes], ranks, samples['OS'][indexes])

with open(cwd / 'figures' / 'survival' / 'results' / 'tmb_weights.pkl', 'wb') as f:
    pickle.dump([test_idx, weights, cancer_test_indexes, cancer_test_ranks, concordances], f)

