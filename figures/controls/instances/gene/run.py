import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from model import DatasetsUtils
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')


from sklearn.metrics import confusion_matrix
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

D, maf = pickle.load(open(cwd / 'figures' / 'controls' / 'data' / 'data.pkl', 'rb'))
maf['Hugo_Symbol'] = maf['Hugo_Symbol'].astype('category')
D['genes'] = np.concatenate(maf[['Hugo_Symbol']].apply(lambda x: x.cat.codes).values + 1)
input_dim = max(D['genes'])
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
y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)
genes_loader = DatasetsUtils.Map.FromNumpy(D['genes'], tf.int16)

y_strat = np.argmax(y_label, axis=-1)

class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)


weights = []
test_idx = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_CE', min_delta=0, patience=20, mode='min', restore_best_weights=True)]
idx_train, idx_test = list(StratifiedShuffleSplit(n_splits=1, test_size=200000, random_state=0).split(y_strat, y_strat))[0]
idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=400000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=600000, ds_size=len(idx_train)))

ds_train = ds_train.map(lambda x: ((genes_loader(x),),
                                   (y_label_loader(x),)))

ds_valid = tf.data.Dataset.from_tensor_slices(((D['genes'][idx_valid]
                                               ),
                                               (
                                                   y_label[idx_valid],
                                               )))

ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=True)


losses = [Losses.CrossEntropy()]
while True:
    gene_encoder = InstanceModels.GeneEmbed(shape=(), input_dim=input_dim, dim=128, regularization=0)
    mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[gene_encoder.model], output_dims=[y_label.shape[-1]], mil_hidden=[32, 16], mode='none')
    mil.model.compile(loss=losses,
                      metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    mil.model.fit(ds_train,
                  steps_per_epoch=10,
                  epochs=1100,
                  validation_data=ds_valid,
                  shuffle=True,
                  callbacks=callbacks)

    eval = mil.model.evaluate(ds_valid)
    if eval[1] < .00005:
        break

with open(cwd / 'figures' / 'controls' / 'instances' / 'gene' / 'results' / 'weights.pkl', 'wb') as f:
    pickle.dump(mil.model.get_weights(), f)


ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
ds_test = ds_test.batch(len(idx_test), drop_remainder=True)
ds_test = ds_test.map(lambda x, y: ((tf.gather(D['genes'], x),
                                       ),
                                         y,
                                         ))

mil.model.evaluate(ds_test)

P = mil.model.predict(ds_test)
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)

y_true = np.argmax(y_label[idx_test], axis=-1)
y_pred = np.argmax(predictions, axis=-1)

matrix = confusion_matrix(y_true, y_pred)

with open(cwd / 'figures' / 'controls' / 'instances' / 'gene' / 'results' / 'matrix.pkl', 'wb') as f:
    pickle.dump(matrix, f)



