import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[3], True)
tf.config.experimental.set_visible_devices(physical_devices[3], 'GPU')

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

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

hist_emb_mat = np.concatenate([np.zeros(samples['histology'].shape[1])[np.newaxis, :], np.diag(np.ones(samples['histology'].shape[1]))], axis=0)
samples['hist_emb'] = hist_emb_mat[np.argmax(samples['histology'], axis=-1)]

##bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])


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

pos_loc = np.array([D['pos_loc'][i] for i in indexes], dtype='object')
pos_bin = np.array([D['pos_bin'][i] for i in indexes], dtype='object')
chr = np.array([D['chr'][i] for i in indexes], dtype='object')

pos_loader = DatasetsUtils.Map.FromNumpy(pos_loc, tf.float32)
bin_loader = DatasetsUtils.Map.FromNumpy(pos_bin, tf.float32)
chr_loader = DatasetsUtils.Map.FromNumpy(chr, tf.int32)

ones_loader = DatasetsUtils.Map.FromNumpy(np.array([np.ones_like(D['pos_loc'])[i] for i in indexes], dtype='object'), tf.float32)


# set y label
y_label = np.log(sample_df['non_syn_counts'].values / (panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]
y_strat = np.argmax(samples['histology'], axis=-1)

runs = 3
initial_weights = []
losses = [Losses.QuantileLoss()]
metrics = [Metrics.QuantileLoss()]
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_QL', min_delta=0.0001, patience=40, mode='min', restore_best_weights=True)]

##for sequence
for i in range(runs):
    # pass_encoder = InstanceModels.PassThrough(shape=(1,))
    # position_encoder = InstanceModels.VariantPositionBin(24, 100)
    sequence_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8], fusion_dimension=32)
    # mil = RaggedModels.MIL(instance_encoders=[pass_encoder.model], output_dim=1, pooling='sum', mil_hidden=(64, 32, 16), output_type='quantiles', regularization=0)
    # mil = RaggedModels.MIL(instance_encoders=[position_encoder.model], output_dim=1, pooling='sum', mil_hidden=(64, 32, 16), output_type='quantiles', regularization=0)
    mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], output_dim=1, pooling='sum', mil_hidden=(64, 32, 16), output_type='quantiles', regularization=0)

    mil.model.compile(loss=losses,
                      metrics=metrics,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    initial_weights.append(mil.model.get_weights())

weights = []
##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1500, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]


    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train], y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=512, ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x, y: ((
                                           # ones_loader(x, ragged_output=True)
                                           # pos_loader(x, ragged_output=True),
                                           # bin_loader(x, ragged_output=True),
                                           # chr_loader(x, ragged_output=True)
                                           five_p_loader(x, ragged_output=True),
                                           three_p_loader(x, ragged_output=True),
                                           ref_loader(x, ragged_output=True),
                                           alt_loader(x, ragged_output=True),
                                           strand_loader(x, ragged_output=True)
                                           ),
                                           y,
                                           ))



    ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid]))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
    ds_valid = ds_valid.map(lambda x, y: ((
                                           # ones_loader(x, ragged_output=True)
                                           # pos_loader(x, ragged_output=True),
                                           # bin_loader(x, ragged_output=True),
                                           # chr_loader(x, ragged_output=True)
                                           five_p_loader(x, ragged_output=True),
                                           three_p_loader(x, ragged_output=True),
                                           ref_loader(x, ragged_output=True),
                                           alt_loader(x, ragged_output=True),
                                           strand_loader(x, ragged_output=True)
                                           ),
                                           y,
                                           ))
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


with open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_341' / 'results' / 'run_sequence.pkl', 'wb') as f:
    pickle.dump(weights, f)


