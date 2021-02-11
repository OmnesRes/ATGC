import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')


from sklearn.metrics import confusion_matrix
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, maf = pickle.load(open(cwd / 'figures' / 'controls' / 'data' / 'data.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

##bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])


maf['label'] = np.zeros(maf.shape[0])
maf['label'][maf['Hugo_Symbol'] == 'MLH1'] = 1
maf['label'][maf['Hugo_Symbol'] == 'MSH2'] = 2
maf['label'][maf['Hugo_Symbol'] == 'MSH6'] = 3
maf['label'][maf['Hugo_Symbol'] == 'PMS2'] = 4
maf['label'][maf['Hugo_Symbol'] == 'TP53'] = 5
maf['label'][maf['Hugo_Symbol'] == 'PTEN'] = 6
maf['label'][maf['Hugo_Symbol'] == 'KRAS'] = 7

A = maf['label'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]

y_label = classes_onehot
y_strat = np.argmax(y_label, axis=-1)

class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)



weights = []
test_idx = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_CE', min_delta=0.00001, patience=50, mode='min', restore_best_weights=True)]
for idx_train, idx_test in StratifiedKFold(n_splits=2, random_state=0).split(y_strat, y_strat):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=500000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train]))
    ds_train = ds_train.batch(len(idx_train), drop_remainder=True)
    ds_train = ds_train.map(lambda x, y: ((tf.gather(tf.constant(D['pos_loc'], dtype=tf.float32), x),
                                              tf.gather(tf.constant(D['pos_bin'], dtype=tf.float32), x),
                                              tf.gather(tf.constant(D['chr'], dtype=tf.int32), x),
                                              ),
                                             y,
                                             ))


    ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid]))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=True)
    ds_valid = ds_valid.map(lambda x, y: ((tf.gather(tf.constant(D['pos_loc'], dtype=tf.float32), x),
                                              tf.gather(tf.constant(D['pos_bin'], dtype=tf.float32), x),
                                              tf.gather(tf.constant(D['chr'], dtype=tf.int32), x),
                                              ),
                                             y,
                                             ))

    ds_train = iter(ds_train).get_next()
    ds_valid_batch = iter(ds_valid).get_next()

    losses = [Losses.CrossEntropy()]

    while True:
        position_encoder = InstanceModels.VariantPositionBin(24, 100)
        mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[position_encoder.model], output_dim=y_label.shape[-1], output_type='anlulogits', mil_hidden=[32, 16], mode='none')
        mil.model.compile(loss=losses,
                          metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,
                                                             ))

        mil.model.fit(x=ds_train[0],
                      y=ds_train[1],
                      batch_size=len(idx_train) // 4,
                      epochs=1000000,
                      validation_data=ds_valid_batch,
                      shuffle=True,
                      callbacks=callbacks)


        eval = mil.model.evaluate(ds_valid)
        if eval[1] < .0005:
            break

    weights.append(mil.model.get_weights())
    test_idx.append(idx_test)


with open('figures/controls/instances/position/results/weights.pkl', 'wb') as f:
    pickle.dump(weights, f)


predictions = []
for weight, idx_test in zip(weights, test_idx):
    mil.model.set_weights(weight)
    ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=True)
    ds_test = ds_test.map(lambda x, y: ((tf.gather(tf.constant(D['pos_loc'], dtype=tf.float32), x),
                                              tf.gather(tf.constant(D['pos_bin'], dtype=tf.float32), x),
                                              tf.gather(tf.constant(D['chr'], dtype=tf.int32), x),
                                              ),
                                             y,
                                             ))

    mil.model.evaluate(ds_test)

    P = mil.model.predict(ds_test)
    z = np.exp(P - np.max(P, axis=1, keepdims=True))
    predictions.append(z / np.sum(z, axis=1, keepdims=True))

y_true = np.argmax(y_label[np.concatenate(test_idx)], axis=-1)
y_pred = np.argmax(np.concatenate(predictions, axis=0), axis=-1)

matrix = confusion_matrix(y_true, y_pred)

with open(cwd / 'figures' / 'controls' / 'instances' / 'position' / 'results' / 'matrix.pkl', 'wb') as f:
    pickle.dump(matrix, f)



