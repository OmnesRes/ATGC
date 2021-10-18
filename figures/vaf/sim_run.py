import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels, SampleModels
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pickle
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

##load the instance and sample data
tables, purities = pickle.load(open(cwd / 'figures' / 'vaf' / 'sim_data.pkl', 'rb'))

ref_reads = np.array([i.ref_counts.values[:, np.newaxis] for i in tables], dtype='object') / np.mean(np.concatenate([i.ref_counts.values for i in tables]))
alt_reads = np.array([i.var_counts.values[:, np.newaxis] for i in tables], dtype='object') / np.mean(np.concatenate([i.var_counts.values for i in tables]))

ref_loader = DatasetsUtils.Map.FromNumpy(ref_reads, tf.float32)
alt_loader = DatasetsUtils.Map.FromNumpy(alt_reads, tf.float32)


y_label_0 = np.array([len(i.clone.unique()) for i in tables])
y_emb_matrix = np.diag(np.ones(max(y_label_0)))
y_label_0 = y_emb_matrix[y_label_0 - 1]
y_strat = np.argmax(y_label_0, axis=-1)
y_label_1 = np.array(purities)[:, np.newaxis]

idx_train, idx_test = next(StratifiedShuffleSplit(random_state=0, n_splits=1, test_size=1000).split(y_strat, y_strat))
idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

with open(cwd / 'figures' / 'vaf' / 'idx_test.pkl', 'wb') as f:
    pickle.dump(idx_test, f)

ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=200, ds_size=len(idx_train)))

ds_train = ds_train.map(lambda x: ((
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                      ),
                                      (tf.gather(tf.constant(y_label_0), x),
                                       tf.gather(tf.constant(y_label_1), x)
                                       )))

ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid))
ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
ds_valid = ds_valid.map(lambda x: ((
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                       ),
                                   (tf.gather(tf.constant(y_label_0), x),
                                    tf.gather(tf.constant(y_label_1), x)
                                    )
                                    ))

ds_test = tf.data.Dataset.from_tensor_slices((idx_test))
ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
ds_test = ds_test.map(lambda x: ((
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                      ),
                                 (tf.gather(tf.constant(y_label_0), x),
                                  tf.gather(tf.constant(y_label_1), x)
                                  )
                                 ))

eval = 100
for i in range(3):
    tile_encoder = InstanceModels.Reads(fused_layers=[32, 64, 128])
    mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model],
                           output_dims=[y_label_0.shape[-1], y_label_1.shape[-1]],
                           output_types=['classification', 'anlulogits'],
                           pooling='dynamic',
                           attention_layers=[32, 16],
                           mil_hidden=[128, 64, 32, 16],
                           regularization=0
                           )
    losses = [tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              'mse']
    # losses = ['mse']
    mil.model.compile(loss=losses,
                      loss_weights=[1, 100],
                      metrics={'output_0': 'accuracy'},
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                         clipvalue=1000))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=30, mode='min', restore_best_weights=True)]
    history = mil.model.fit(ds_train, steps_per_epoch=40, validation_data=ds_valid, epochs=10000, callbacks=callbacks)
    run_eval = mil.model.evaluate(ds_valid)[0]
    if run_eval < eval:
        eval = run_eval
        run_history = history
        run_evaluation = mil.model.evaluate(ds_test)
        run_weights = mil.model.get_weights()



mil.model.set_weights(run_weights)
predictions = mil.model.predict(ds_test)
attentions = mil.attention_model.predict(ds_test).to_list()


with open(cwd / 'figures' / 'vaf' / 'atgc_predictions.pkl', 'wb') as f:
    pickle.dump([run_weights, predictions, attentions], f)
