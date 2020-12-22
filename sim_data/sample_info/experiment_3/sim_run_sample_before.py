import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels, SampleModels
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit
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

##load the instance and sample data
D, samples = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_3' / 'sim_data.pkl', 'rb'))

##perform embeddings with a zero vector for index 0
strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

indexes = [np.where(D['sample_idx'] == idx) for idx in range(len(samples['type']))]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')
instance_type = np.array([np.array(samples['type'])[D['sample_idx']][i] for i in indexes], dtype='object')

five_p_loader = DatasetsUtils.Map.FromNumpy(five_p, tf.int32)
three_p_loader = DatasetsUtils.Map.FromNumpy(three_p, tf.int32)
ref_loader = DatasetsUtils.Map.FromNumpy(ref, tf.int32)
alt_loader = DatasetsUtils.Map.FromNumpy(alt, tf.int32)
strand_loader = DatasetsUtils.Map.FromNumpy(strand, tf.float32)
type_loader = DatasetsUtils.Map.FromNumpy(instance_type, tf.int32)


y_label = np.log(np.array(samples['values']) + 1)[:, np.newaxis]
y_strat = np.array(samples['type'])


idx_train, idx_test = next(StratifiedShuffleSplit(random_state=0, n_splits=1, test_size=200).split(y_strat, y_strat))
idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]


ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train], y_strat[idx_train]))
ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=100, ds_size=len(idx_train)))
ds_train = ds_train.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                       three_p_loader(x, ragged_output=True),
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                       strand_loader(x, ragged_output=True),
                                       type_loader(x, ragged_output=True)
                                       ),
                                       y))

ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid]))
ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
ds_valid = ds_valid.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                       three_p_loader(x, ragged_output=True),
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                       strand_loader(x, ragged_output=True),
                                       type_loader(x, ragged_output=True)
                                       ),
                                       y))

ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
ds_test = ds_test.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                       three_p_loader(x, ragged_output=True),
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                       strand_loader(x, ragged_output=True),
                                       type_loader(x, ragged_output=True)
                                       ),
                                       y))

histories = []
evaluations = []
weights = []
for i in range(3):
    sequence_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])
    class_encoder = SampleModels.Type(shape=(), dim=10)
    mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model, class_encoder.model], instance_layers=[64, ], output_dim=1, pooling='sum', output_type='other')
    losses = ['mse']
    mil.model.compile(loss=losses,
                      metrics=['mse'],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                    ))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_mse', min_delta=0.001, patience=20, mode='min', restore_best_weights=True)]
    history = mil.model.fit(ds_train, steps_per_epoch=10, validation_data=ds_valid, epochs=10000, callbacks=callbacks)
    evaluation = mil.model.evaluate(ds_test)
    histories.append(history.history)
    evaluations.append(evaluation)
    weights.append(mil.model.get_weights())


with open(cwd / 'sim_data' / 'sample_info' / 'experiment_3' / 'sample_model_sum_before.pkl', 'wb') as f:
    pickle.dump([evaluations, histories, weights], f)