import numpy as np
from model.Sample_MIL import RaggedModels, InstanceModels
from model.KerasLayers import Losses, Metrics
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from model import DatasetsUtils
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-4], True)
tf.config.experimental.set_visible_devices(physical_devices[-4], 'GPU')

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'pcawg' / 'data' / 'data.pkl', 'rb'))
contexts = tcga_maf['contexts'].astype('category').cat.codes.values
del tcga_maf
class_counts = dict(samples['histology'].value_counts())
indexes = [np.where(D['sample_idx'] == idx) for idx in samples.index]
del D

contexts = np.array([contexts[i] for i in indexes], dtype='object')
dropout = .6
index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=dropout)
context_loader = DatasetsUtils.Map.FromNumpyandIndices(contexts, tf.int16)

context_loader_eval = DatasetsUtils.Map.FromNumpy(contexts, tf.int16)

A = samples['histology'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot

y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)
y_weights_loader = DatasetsUtils.Map.FromNumpy(y_weights, tf.float32)

batch_size = 256
test_idx = []
weights = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='weighted_CE', min_delta=0.01, patience=40, mode='min', restore_best_weights=True)]
losses = [Losses.CrossEntropy()]
for idx_train, idx_test in StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(y_strat, y_strat):
    eval = 100
    test_idx.append(idx_test)

    with tf.device('/cpu:0'):
        ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
        ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=batch_size, ds_size=len(idx_train)))

        ds_train = ds_train.map(lambda x: ((
            index_loader(x),
        )

        ),
                                )

        ds_train = ds_train.map(lambda x: ((
                                                context_loader(x[0], x[1]),

                                               ),
                                              (
                                                  y_label_loader(x[0]),
                                              ),
                                               y_weights_loader(x[0])
        ),
                                )

        ds_train.prefetch(1)


    for i in range(3):

        context_encoder = InstanceModels.Type(shape=(), dim=97)
        mil = RaggedModels.MIL(instance_encoders=[context_encoder.model], sample_encoders=[], heads=1, output_dims=[y_label.shape[-1]], mil_hidden=[1024, 1024, 512], attention_layers=[], dropout=.5, instance_dropout=0, regularization=0, input_dropout=0, mode='attention', pooling='dynamic', dynamic_hidden=[128, 32])

        mil.model.compile(loss=losses,
                          metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
                          weighted_metrics=[Metrics.CrossEntropy()],
                          optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001,
                                                             ))

        mil.model.fit(ds_train,
                      steps_per_epoch=(len(idx_train) // batch_size + 1) * 4,
                      epochs=20000,
                      callbacks=callbacks,
                      )

        run_eval = mil.model.evaluate(ds_train, steps=200)[-1]

        if run_eval < eval:
            eval = run_eval
            run_weights = mil.model.get_weights()

    weights.append(run_weights)


with open(cwd / 'figures' / 'tumor_classification' / 'pcawg' / 'mil_encoder' / 'results' / 'context_weights.pkl', 'wb') as f:
    pickle.dump([test_idx, weights], f)

context_encoder = InstanceModels.Type(shape=(), dim=97)
mil = RaggedModels.MIL(instance_encoders=[context_encoder.model], sample_encoders=[], heads=1, output_dims=[y_label.shape[-1]], mil_hidden=[1024, 1024, 512], attention_layers=[], dropout=.5, instance_dropout=0, regularization=0, input_dropout=dropout, mode='attention', pooling='dynamic', dynamic_hidden=[128, 32])

predictions = []
for weight, idx_test in zip(weights, test_idx):
    mil.model.set_weights(weight)
    with tf.device('/cpu:0'):
        ds_test = tf.data.Dataset.from_tensor_slices(((
                                                   context_loader_eval(idx_test),
                                               ),
                                                (
                                                    tf.gather(y_label, idx_test),
                                                ),
                                                tf.gather(y_weights, idx_test)
                                                ))
        ds_test = ds_test.batch(1, drop_remainder=False)
    predictions.append(mil.model.predict(ds_test))

P = np.concatenate(predictions)
#convert the model logits to probablities
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)
#
with open(cwd / 'figures' / 'tumor_classification' / 'pcawg' / 'mil_encoder' / 'results' / 'context_predictions.pkl', 'wb') as f:
    pickle.dump([predictions, y_label, test_idx], f)

print(np.sum((np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) * y_weights[np.concatenate(test_idx)]))
print(sum(np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) / len(y_label))
print(roc_auc_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), predictions, multi_class='ovr'))

# 0.8328172193678938
# 0.8626790227464195
# 0.9860024120359953



