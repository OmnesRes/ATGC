import numpy as np
import pandas as pd
import tensorflow as tf
from model_new.KerasModels import InstanceModels, RaggedModels, SampleModels
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[3], True)
tf.config.experimental.set_visible_devices(physical_devices[3], 'GPU')

import pathlib
cwd = pathlib.PosixPath('/home/janaya2/Desktop/ATGC2')
# path = pathlib.Path.cwd()
# if path.stem == 'ATGC2':
#     cwd = path
# else:
#     cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
#     import sys
#     sys.path.append(str(cwd))

D, samples, maf, sample_df = pickle.load(open(cwd / 'tmb_tests' / 'data.pkl', 'rb'))
panels = pickle.load(open(cwd / '..' / 'ATGC_paper' / 'files' / 'tcga_panel_table.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

indexes = np.argsort(D['sample_idx'])

five_p = tf.RaggedTensor.from_value_rowids(D['seq_5p'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=sample_df.shape[0])
three_p = tf.RaggedTensor.from_value_rowids(D['seq_3p'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=sample_df.shape[0])
ref = tf.RaggedTensor.from_value_rowids(D['seq_ref'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=sample_df.shape[0])
alt = tf.RaggedTensor.from_value_rowids(D['seq_alt'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=sample_df.shape[0])
strand = tf.RaggedTensor.from_value_rowids(D['strand_emb'][indexes].astype(np.float32), D['sample_idx'][indexes], nrows=sample_df.shape[0])

hist_emb_mat = np.concatenate([np.zeros(samples['histology'].shape[1])[np.newaxis, :], np.diag(np.ones(samples['histology'].shape[1]))], axis=0)
samples['hist_emb'] = hist_emb_mat[np.argmax(samples['histology'], axis=-1)]

##bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])
ones = tf.RaggedTensor.from_value_rowids(np.ones_like(D['pos_loc'])[indexes].astype(np.float32), D['sample_idx'][indexes], nrows=sample_df.shape[0])


tile_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])


# set y label
y_label = np.log(sample_df['non_syn_counts'].values / (panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]
y_strat = np.argmax(samples['histology'], axis=-1)

runs = 3
initial_weights = []
# metrics = [RaggedModels.losses.QuantileLoss()]
# losses = [RaggedModels.losses.QuantileLoss()]
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_mse', min_delta=0.0001, patience=40, mode='min', restore_best_weights=True)]

for i in range(runs):
    mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model], sample_encoders=[], output_dim=1, regularization=False, output_type='quantiles')
    mil.model.compile(loss='mse', metrics=['mse'])
    initial_weights.append(mil.model.get_weights())

weights = []
##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat):

    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1500, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    train_data = (tf.gather(five_p, idx_train), tf.gather(three_p, idx_train), tf.gather(ref, idx_train), tf.gather(alt, idx_train), tf.gather(strand, idx_train))
    valid_data = (tf.gather(five_p, idx_valid), tf.gather(three_p, idx_valid), tf.gather(ref, idx_valid), tf.gather(alt, idx_valid), tf.gather(strand, idx_valid))

    tfds_train = tf.data.Dataset.from_tensor_slices((train_data, y_label[idx_train]))
    tfds_train = tfds_train.shuffle(len(y_label), reshuffle_each_iteration=True).batch(512, drop_remainder=True)

    tfds_valid = tf.data.Dataset.from_tensor_slices((valid_data, y_label[idx_valid]))
    tfds_valid = tfds_valid.batch(len(idx_valid), drop_remainder=False)

    eval = 100
    for initial_weight in initial_weights:
        mil.model.set_weights(initial_weight)
        mil.model.fit(tfds_train, validation_data=tfds_valid,
                      epochs=10000,
                      callbacks=callbacks)
        run_eval = mil.model.evaluate(tfds_valid)[1]
        if run_eval < eval:
            eval = run_eval
            run_weights = mil.model.get_weights()

    weights.append(run_weights)


with open('tmb_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)