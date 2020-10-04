'''This script will perform a classification task on the data generated in sim_data.py.
Each positive sample has approximately half it's variants a specific sequence.
It is a simple task so should quickly achieve perfect accuracy unless you start with bad weights.
Note: the loss will be much higher than the cross entropy loss due to regularization.'''

##perform imports and set the GPU you want to use
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
disable_eager_execution()
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

from model_old.CustomKerasModels import InputFeatures, ATGC
from model_old.CustomKerasTools import BatchGenerator, Losses

##load the instance and sample data
D, samples = pickle.load(open(cwd / 'sim_data' / 'sim_data.pkl', 'rb'))

##perform embeddings with a zero vector for index 0
strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]


##choose your instance concepts, here a sequence concept of length 6, embedding dim 4, strand 2, and 4 kernels per 5p, 3p, ref, alt.

features = [InputFeatures.VariantSequence(6, 4, 2, [4, 4, 4, 4],
                                          {'5p': D['seq_5p'], '3p': D['seq_3p'], 'ref': D['seq_ref'], 'alt': D['seq_alt'], 'strand': D['strand_emb'], 'cds': D['cds_emb']},
                                          use_frame=False, fusion_dimension=64)
]

##choose your sample concepts
sample_features = ()


# set y label and weights
y_label = np.stack([[0, 1] if i==1 else [1, 0] for i in samples['classes']])
y_strat = np.argmax(y_label, axis=-1)

class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

##build the model
atgc = ATGC(features, sample_features=sample_features, aggregation_dimension=64, fusion_dimension=32)
atgc.build_instance_encoder_model(return_latent=False)
atgc.build_sample_encoder_model()
atgc.build_mil_model(output_dim=y_label.shape[1], output_extra=1, output_type='anlulogits', aggregation='recursion', mil_hidden=(16, 8))
metrics = [Losses.Weighted.CrossEntropyfromlogits.cross_entropy_weighted, Losses.Weighted.Accuracy.accuracy]
atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=10000), loss=Losses.Weighted.CrossEntropyfromlogits.cross_entropy_weighted, metrics=metrics)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_cross_entropy_weighted', min_delta=0.001, patience=5, mode='min', restore_best_weights=True)]
initial_weights = atgc.mil_model.get_weights()

##perform 8 fold stratification
weights = []
test_idxs = []
for idx_train, idx_test in StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=50, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    batch_gen_train = BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach='minibatch', batch_size=32, idx_sample=idx_train)

    data_valid = next(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx_valid).data_generator())


    atgc.mil_model.set_weights(initial_weights)
    atgc.mil_model.fit(batch_gen_train.data_generator(),
                       steps_per_epoch=batch_gen_train.n_splits,
                       epochs=10000,
                       validation_data=data_valid,
                       shuffle=False,
                       callbacks=callbacks)

    weights.append(atgc.mil_model.get_weights())
    test_idxs.append(idx_test)


##check evaluations on the test set for each Kfold
for weight, idx in zip(weights, test_idxs):
    atgc.mil_model.set_weights(weight)
    data_test = next(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx).data_generator())
    print(atgc.mil_model.evaluate(data_test[0], data_test[1]))



