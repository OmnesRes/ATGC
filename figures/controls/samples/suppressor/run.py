import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[7], True)
tf.config.experimental.set_visible_devices(physical_devices[7], 'GPU')

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))


D, maf = pickle.load(open(cwd / 'figures' / 'controls' / 'data' / 'data.pkl', 'rb'))
sample_df = pickle.load(open(cwd / 'files' / 'tcga_sample_table.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

##bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])

indexes = [np.where(D['sample_idx'] == idx) for idx in range(sample_df.shape[0])]

pos_loc = np.array([D['pos_loc'][i] for i in indexes], dtype='object')
pos_bin = np.array([D['pos_bin'][i] for i in indexes], dtype='object')
chr = np.array([D['chr'][i] for i in indexes], dtype='object')


# set y label and weights
genes = maf['Hugo_Symbol'].values
boolean = ['PTEN' in genes[j] for j in [np.where(D['sample_idx'] == i)[0] for i in range(sample_df.shape[0])]]
y_label = np.stack([[0, 1] if i else [1, 0] for i in boolean])
y_strat = np.argmax(y_label, axis=-1)

class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

pos_loader = DatasetsUtils.Map.FromNumpy(pos_loc, tf.float32)
bin_loader = DatasetsUtils.Map.FromNumpy(pos_bin, tf.float32)
chr_loader = DatasetsUtils.Map.FromNumpy(chr, tf.int32)


weights = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_weighted_CE', min_delta=0.0001, patience=20, mode='min', restore_best_weights=True)]
losses = [Losses.CrossEntropy()]
##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train], y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=len(idx_train) // 2, ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x, y: ((pos_loader(x, ragged_output=True),
                                           bin_loader(x, ragged_output=True),
                                           chr_loader(x, ragged_output=True),
                                           ),
                                           y,
                                          tf.gather(tf.constant(y_weights, dtype=tf.float32), x)
                                           ))

    ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid]))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
    ds_valid = ds_valid.map(lambda x, y: ((pos_loader(x, ragged_output=True),
                                           bin_loader(x, ragged_output=True),
                                           chr_loader(x, ragged_output=True),
                                           ),
                                           y,
                                          tf.gather(tf.constant(y_weights, dtype=tf.float32), x)
                                           ))

    while True:
        position_encoder = InstanceModels.VariantPositionBin(24, 100)
        mil = RaggedModels.MIL(instance_encoders=[position_encoder.model], output_dim=2, pooling='both', mil_hidden=(128, 64, 16, 8), output_type='anlulogits')

        mil.model.compile(loss=losses,
                          metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
                          weighted_metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                             clipvalue=10000))
        mil.model.fit(ds_train,
                      steps_per_epoch=20,
                      validation_data=ds_valid,
                      epochs=10000,
                      callbacks=callbacks)


        eval = mil.model.evaluate(ds_valid)
        if eval[2] > .99:
            break
    weights.append(mil.model.get_weights())


with open(cwd / 'figures' / 'controls' / 'samples' / 'suppressor' / 'results' / 'weights.pkl', 'wb') as f:
    pickle.dump(weights, f)


