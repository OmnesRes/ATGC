import numpy as np
import tensorflow as tf
from model.Instance_MIL import InstanceModels, RaggedModels
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[4], True)
tf.config.experimental.set_visible_devices(physical_devices[4], 'GPU')
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

##load the instance and sample data
D, samples = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sim_data.pkl', 'rb'))

##perform embeddings with a zero vector for index 0
strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

indexes = np.argsort(D['sample_idx'])

five_p = tf.RaggedTensor.from_value_rowids(D['seq_5p'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
three_p = tf.RaggedTensor.from_value_rowids(D['seq_3p'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
ref = tf.RaggedTensor.from_value_rowids(D['seq_ref'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
alt = tf.RaggedTensor.from_value_rowids(D['seq_alt'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
strand = tf.RaggedTensor.from_value_rowids(D['strand_emb'][indexes].astype(np.float32), D['sample_idx'][indexes], nrows=len(samples['classes']))

y_label = np.stack([[0, 1] if i == 1 else [1, 0] for i in samples['classes']])
y_strat = np.argmax(y_label, axis=-1)

idx_train, idx_test = next(StratifiedShuffleSplit(random_state=0, n_splits=1, test_size=200).split(y_strat, y_strat))
idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

train_data = (tf.gather(five_p, idx_train), tf.gather(three_p, idx_train), tf.gather(ref, idx_train), tf.gather(alt, idx_train), tf.gather(strand, idx_train))
valid_data = (tf.gather(five_p, idx_valid), tf.gather(three_p, idx_valid), tf.gather(ref, idx_valid), tf.gather(alt, idx_valid), tf.gather(strand, idx_valid))
test_data = (tf.gather(five_p, idx_test), tf.gather(three_p, idx_test), tf.gather(ref, idx_test), tf.gather(alt, idx_test), tf.gather(strand, idx_test))

tfds_train = tf.data.Dataset.from_tensor_slices((train_data, y_label[idx_train]))
tfds_train = tfds_train.shuffle(len(y_label), reshuffle_each_iteration=True).batch(100, drop_remainder=True)

tfds_valid = tf.data.Dataset.from_tensor_slices((valid_data, y_label[idx_valid]))
tfds_valid = tfds_valid.batch(len(idx_valid), drop_remainder=False)

tfds_test = tf.data.Dataset.from_tensor_slices((test_data, y_label[idx_test]))
tfds_test = tfds_test.batch(len(idx_test), drop_remainder=False)

tile_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])

histories = []
evaluations = []
weights = []
for i in range(3):
    mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model], output_dim=2, pooling='sum')
    losses = [tf.keras.losses.CategoricalCrossentropy(from_logits=True)]
    mil.model.compile(loss=losses,
                      metrics=['accuracy', tf.keras.metrics.CategoricalCrossentropy(from_logits=True)],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                    ))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_categorical_crossentropy', min_delta=0.00001, patience=50, mode='min', restore_best_weights=True)]
    history = mil.model.fit(tfds_train, validation_data=tfds_valid, epochs=10000, callbacks=callbacks)
    evaluation = mil.model.evaluate(tfds_test)
    histories.append(history.history)
    evaluations.append(evaluation)
    weights.append(mil.model.get_weights())
    del mil


with open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'instance_model_sum.pkl', 'wb') as f:
    pickle.dump([evaluations, histories, weights], f)