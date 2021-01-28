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
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

from figures.msi.model.MSIModel import InputFeatures, ATGC
from model.CustomKerasTools import BatchGenerator, Losses

D, samples, sample_df = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]


features = [InputFeatures.VariantSequence(20, 4, 2, [8, 8, 8, 8],
                                         {'5p': D['seq_5p'], '3p': D['seq_3p'], 'ref': D['seq_ref'], 'alt': D['seq_alt'], 'strand': D['strand_emb'], 'cds': D['cds_emb']},
                                         use_frame=False)]


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
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_cross_entropy', min_delta=0.0001, patience=20, mode='min', restore_best_weights=True)]
initial_weights = atgc.mil_model.get_weights()

weights = []
##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=9, random_state=0, shuffle=True).split(y_strat, y_strat):

    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    batch_gen_train = BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=None,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach='subsample', batch_size=128, idx_sample=idx_train)

    data_valid = next(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=None,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx_valid).data_generator())


    while True:
        atgc.mil_model.set_weights(initial_weights)
        atgc.mil_model.fit(batch_gen_train.data_generator(),
                           steps_per_epoch=batch_gen_train.n_splits*2,
                           epochs=10000,
                           validation_data=data_valid,
                           shuffle=False,
                           callbacks=callbacks)
        eval = atgc.mil_model.evaluate(data_valid[0], data_valid[1])
        if eval[1] < .07:
            break
        else:
            del atgc
            atgc = ATGC(features, latent_dimension=64)
            atgc.build_instance_encoder_model(return_latent=False)
            atgc.build_mil_model(output_dim=y_label.shape[1], output_extra=1, output_type='classification_probability', aggregation='recursion', mil_hidden=(32, 16))
            atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=10000), loss=Losses.Weighted.CrossEntropy.cross_entropy, metrics=metrics)
            initial_weights = atgc.mil_model.get_weights()

    weights.append(atgc.mil_model.get_weights())


with open(cwd / 'figures' / 'msi' / 'results' / 'run.pkl', 'wb') as f:
    pickle.dump(weights, f)


