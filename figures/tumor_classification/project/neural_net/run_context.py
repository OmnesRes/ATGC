import numpy as np
import pandas as pd
from model.Sample_MIL import InstanceModels, RaggedModels
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
tf.config.experimental.set_memory_growth(physical_devices[-2], True)
tf.config.experimental.set_visible_devices(physical_devices[-2], 'GPU')

D, tcga_maf, samples = pickle.load(open(cwd / 'figures' / 'tumor_classification' / 'data' / 'data.pkl', 'rb'))
tcga_maf = tcga_maf.loc[:, ['Tumor_Sample_Barcode', 'contexts']]

context_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "contexts"]).size().unstack(fill_value=0)
context_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': context_df.index, 'context_counts': context_df.values.tolist()})

samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
class_counts = dict(samples['type'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] > 125]
samples = samples.loc[samples['type'].isin(labels_to_use)]
samples = pd.merge(samples, context_df, on='Tumor_Sample_Barcode', how='left')

A = samples['type'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot

y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)
context_counts = np.apply_along_axis(lambda x: np.log(x + 1), 0, np.stack(samples['context_counts'].values))

contexts_loader = DatasetsUtils.Map.FromNumpy(context_counts, tf.float32)
y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)
y_weights_loader = DatasetsUtils.Map.FromNumpy(y_weights, tf.float32)

predictions = []
evaluations = []
test_idx = []
weights = []
batch_size = 512
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=25, mode='min', restore_best_weights=True)]
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat):
    test_idx.append(idx_test)
    temp_evaluations = []
    eval = 100
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=batch_size, ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x: ((
                                          contexts_loader(x),
                                           ),
                                          (
                                          y_label_loader(x),
                                          ),
                                           y_weights_loader(x)
                                          )
                            )

    ds_valid = tf.data.Dataset.from_tensor_slices((
                                                  (
                                                   context_counts[idx_valid],
                                                   ),
                                                  (
                                                   y_label[idx_valid],
                                                  ),
                                                   y_weights[idx_valid]
                                                   ))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)

    ds_test = tf.data.Dataset.from_tensor_slices((
                                                 (
                                                  context_counts[idx_test],
                                                 ),
                                                 (
                                                  y_label[idx_test],
                                                 ),

                                                  y_weights[idx_test]
                                                  ))

    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)

    losses = [Losses.CrossEntropy()]

    for run in range(3):
        context_encoder = InstanceModels.Feature(shape=(context_counts.shape[-1]), input_dropout=0, layers=[512], regularization=.01, layer_dropouts=[.3])
        mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[context_encoder.model], output_dims=[y_label.shape[-1]], output_types=['other'], mil_hidden=[256, 128], mode='none', dropout=.3)
        mil.model.compile(loss=losses,
                          metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                             ))
        mil.model.fit(ds_train,
                      steps_per_epoch=10,
                      epochs=20000,
                      validation_data=ds_valid,
                      callbacks=callbacks)
        run_eval = mil.model.evaluate(ds_valid)[0]
        temp_evaluations.append(run_eval)
        if run_eval < eval:
            eval = run_eval
            run_weights = mil.model.get_weights()
            print('test_eval', mil.model.evaluate(ds_test))
    mil.model.set_weights(run_weights)
    predictions.append(mil.model.predict(ds_test))
    weights.append(run_weights)
    evaluations.append(temp_evaluations)


with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'context_weights.pkl', 'wb') as f:
    pickle.dump([test_idx, weights], f)

P = np.concatenate(predictions)
#convert the model logits to probablities
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)

with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'context_predictions.pkl', 'wb') as f:
    pickle.dump([test_idx, predictions], f)

print(np.sum((np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) * y_weights[np.concatenate(test_idx)]))
print(sum(np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) / len(y_label))
print(roc_auc_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), predictions, multi_class='ovr'))

# 0.5145933693273526
# 0.4987819086961127
# 0.9337750272141737