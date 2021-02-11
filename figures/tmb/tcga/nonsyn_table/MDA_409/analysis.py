import numpy as np
import tensorflow as tf
import pandas as pd
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
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


D, samples, maf, sample_df = pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'nonsyn_table' / 'MDA_409' / 'data' / 'data.pkl', 'rb'))
panels = pickle.load(open(cwd / 'files' / 'tcga_panel_table.pkl', 'rb'))


##bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])

indexes = [np.where(D['sample_idx'] == idx) for idx in range(sample_df.shape[0])]

ones_loader = DatasetsUtils.Map.FromNumpy(np.array([np.ones_like(D['pos_loc'])[i] for i in indexes], dtype='object'), tf.float32)


loaders = [
    [ones_loader],
]


# set y label
y_label = np.log(sample_df['non_syn_counts'].values/(panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]
y_strat = np.argmax(samples['histology'], axis=-1)

losses = [Losses.QuantileLoss()]
metrics = [Metrics.QuantileLoss()]

encoders = [InstanceModels.PassThrough(shape=(1,)),
           ]

all_weights = [
    pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'nonsyn_table' / 'MDA_409' / 'results' / 'run_naive.pkl', 'rb'))
    ]

results = {}

for encoder, loaders, weights, name in zip(encoders, loaders, all_weights, ['naive']):

    mil = RaggedModels.MIL(instance_encoders=[encoder.model], output_dim=1, pooling='sum', mil_hidden=(64, 32, 16), output_type='quantiles', regularization=0)
    mil.model.compile(loss=losses,
                      metrics=metrics,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    ##test eval
    test_idx = []
    predictions = []

    for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat)):
        mil.model.set_weights(weights[index])

        ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
        ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
        ds_test = ds_test.map(lambda x, y: (tuple([i(x, ragged_output=True) for i in loaders]),
                                            y,
                                            ))
        predictions.append(mil.model.predict(ds_test))
        test_idx.append(idx_test)

    #mse
    print(round(np.mean((y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])**2), 4))
    #mae
    print(round(np.mean(np.absolute(y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])), 4))
    #r2
    print(round(r2_score(y_label[:, 0][np.concatenate(test_idx)], np.concatenate(predictions)[:, 1]), 4))
    print()
    results[name] = np.concatenate(predictions)