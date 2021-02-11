import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels, SampleModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pickle
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


D, samples, maf, sample_df = pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_341' / 'data' / 'data.pkl', 'rb'))
panels = pickle.load(open(cwd / 'files' / 'tcga_panel_table.pkl', 'rb'))

counts = np.array([len(np.where(D['sample_idx'] == idx)[0]) for idx in range(sample_df.shape[0])])[:, np.newaxis]

# set y label
y_label = np.log(sample_df['non_syn_counts'].values / (panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]

##if using tcga cancer types
# y_strat = np.argmax(samples['histology'], axis=-1)

##if using NCI-T labels
label_counts = sample_df['NCI-T Label'].value_counts().to_dict()
mask = sample_df['NCI-T Label'].map(lambda x: label_counts.get(x, 0) >= 36)
y_label = y_label[mask]
counts = counts[mask]
labels = [i for i in sorted(label_counts.keys()) if label_counts[i] >= 36]
y_strat = sample_df['NCI-T Label'][mask].map(lambda x: labels.index(x)).values


runs = 3
initial_weights = []
losses = [Losses.QuantileLoss()]
metrics = [Metrics.QuantileLoss()]
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_QL', min_delta=0.0001, patience=40, mode='min', restore_best_weights=True)]

##for sequence
for i in range(runs):
    pass_encoder = InstanceModels.PassThrough(shape=(1,))
    type_encoder = SampleModels.Type(shape=(), dim=max(y_strat) + 1)
    mil = RaggedModels.MIL(sample_encoders=[pass_encoder.model, type_encoder.model], output_dim=1, mil_hidden=(64, 32, 16), output_type='quantiles', regularization=0, mode='none')

    mil.model.compile(loss=losses,
                      metrics=metrics,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    initial_weights.append(mil.model.get_weights())


def extend_data(x):
    return (tf.repeat(x[0], 2, axis=0), tf.reshape(tf.stack([x[-1], tf.zeros_like(x[-1])], axis=-1), [tf.shape(x[0])[0] * 2, ]))


weights = []
##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1500, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]


    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train], y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=512, ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x, y: ((
                                           tf.gather(tf.constant(counts), x),
                                           tf.gather(tf.constant(y_strat + 1), x)
                                            ),
                                           y,
                                           ))
    ds_train = ds_train.map(lambda x, y: (extend_data(x), tf.repeat(y, 2, axis=0)))


    ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid]))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
    ds_valid = ds_valid.map(lambda x, y: ((
                                           tf.gather(tf.constant(counts), x),
                                           tf.gather(tf.constant(y_strat + 1), x)),
                                           y,
                                           ))

    ds_valid = ds_valid.map(lambda x, y: (extend_data(x), tf.repeat(y, 2, axis=0)))

    eval = 100
    for initial_weight in initial_weights:
        mil.model.set_weights(initial_weight)
        mil.model.fit(ds_train,
                      steps_per_epoch=10,
                      validation_data=ds_valid,
                      epochs=10000,
                      callbacks=callbacks,
                      workers=1)

        run_eval = mil.model.evaluate(ds_valid)[1]
        if run_eval < eval:
            eval = run_eval
            run_weights = mil.model.get_weights()

    weights.append(run_weights)


with open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_341' / 'results' / 'run_naive_sample_nci.pkl', 'wb') as f:
    pickle.dump(weights, f)


