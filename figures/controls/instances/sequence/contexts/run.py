import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
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

from model.CustomKerasModels import InputFeatures, ATGC
from model.CustomKerasTools import BatchGenerator, Losses

D, maf = pickle.load(open(cwd / 'figures' / 'controls' / 'data' / 'data.pkl', 'rb'))


variant_labels = maf.contexts.astype('category').cat.codes
D['contexts'] = np.eye(97)[variant_labels]

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

features = [InputFeatures.VariantSequence(6, 4, 2, [64, 64, 64, 64],
                                          {'5p': D['seq_5p'], '3p': D['seq_3p'], 'ref': D['seq_ref'], 'alt': D['seq_alt'], 'strand': D['strand_emb']},
                                          use_frame=False, fusion_dimension=128)]

sample_features = []

# set y label and weights
y_strat = np.arange(97)[np.argwhere(D['contexts'])[:, 1]]
y_label = D['contexts']
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

# set train, valid, test indices - with stratification if needed
idx_train, idx_test = list(StratifiedShuffleSplit(n_splits=1, test_size=200000, random_state=0).split(y_strat, y_strat))[0]
idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]


atgc = ATGC(features, aggregation_dimension=128, fusion_dimension=128)
atgc.build_instance_encoder_model(return_latent=False)
atgc.build_mil_model(output_dim=y_label.shape[1], output_extra=1, output_type='logits', aggregation='none', mil_hidden=())
metrics = [Losses.Weighted.CrossEntropyfromlogits.cross_entropy, Losses.Weighted.CrossEntropyfromlogits.cross_entropy_weighted, Losses.Weighted.Accuracy.accuracy, Losses.Weighted.Accuracy.accuracy_weighted]
atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=Losses.Weighted.CrossEntropyfromlogits.cross_entropy_weighted, metrics=metrics)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_cross_entropy', min_delta=0.001, patience=10, mode='min', restore_best_weights=True)]

batch_gen_train = BatchGenerator(x_instance_sample_idx=None, x_instance_features=features, x_sample=sample_features,
                                 y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach='subsample', batch_size=50000, idx_sample=idx_train)

data_valid = next(BatchGenerator(x_instance_sample_idx=None, x_instance_features=features, x_sample=sample_features,
                                 y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx_valid).data_generator())

data_test = next(BatchGenerator(x_instance_sample_idx=None, x_instance_features=features, x_sample=sample_features,
                                y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx_test).data_generator())

atgc.mil_model.fit(batch_gen_train.data_generator(),
                   steps_per_epoch=20,
                   epochs=100000,
                   validation_data=data_valid,
                   shuffle=False,
                   callbacks=callbacks)

with open(cwd / 'figures' / 'controls' / 'instances' / 'sequence' / 'contexts' / 'results' / 'weights.pkl', 'wb') as f:
    pickle.dump(atgc.mil_model.get_weights(), f)


P = atgc.mil_model.predict(data_test[0])
P = P[0, :, : -1]
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)

y_true = np.argmax(y_label[idx_test], axis=-1)
y_pred = np.argmax(predictions, axis=-1)

matrix = confusion_matrix(y_true, y_pred)

with open(cwd / 'figures' / 'controls' / 'instances' / 'sequence' / 'contexts' / 'results' / 'weights.pkl', 'wb') as f:
    pickle.dump(matrix, f)
