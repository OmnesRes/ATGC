import numpy as np
import pandas as pd
from model.KerasLayers import Losses, Metrics
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from model import DatasetsUtils
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
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
tcga_maf = tcga_maf.loc[:, ['Tumor_Sample_Barcode', 'Hugo_Symbol']]

gene_df = tcga_maf.groupby(['Tumor_Sample_Barcode', "Hugo_Symbol"]).size().unstack(fill_value=0)
gene_df = pd.DataFrame.from_dict({'Tumor_Sample_Barcode': gene_df.index, 'gene_counts': gene_df.values.tolist()})

samples['type'] = samples['type'].apply(lambda x: 'COAD' if x == 'READ' else x)
class_counts = dict(samples['type'].value_counts())
labels_to_use = [i for i in class_counts if class_counts[i] > 125]
samples = samples.loc[samples['type'].isin(labels_to_use)]
samples = pd.merge(samples, gene_df, on='Tumor_Sample_Barcode', how='left')

A = samples['type'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_label = classes_onehot

y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

gene_counts = np.apply_along_axis(lambda x: np.log(x + 1), 0, np.stack(samples['gene_counts'].values))

gene_loader = DatasetsUtils.Map.FromNumpy(gene_counts, tf.float32)
y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)
y_weights_loader = DatasetsUtils.Map.FromNumpy(y_weights, tf.float32)


dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
dim_weight_decay = Real(low=1e-3, high=0.5, prior = 'log-uniform', name='weight_decay')
dim_num_dense_layers = Integer(low=0, high=3, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'softplus'], name='activation')
dim_dropout = Real(low=1e-6, high=0.5, prior='log-uniform', name='dropout')
dimensions = [dim_learning_rate, dim_weight_decay, dim_dropout, dim_num_dense_layers, dim_num_dense_nodes, dim_activation]
default_paramaters = [1e-4, 1e-3, 1e-6, 0, 100, 'relu']

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_weighted_CE', min_delta=0.01, patience=10, mode='min', restore_best_weights=True)]
losses = [Losses.CrossEntropy()]

def create_model(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation):
    ###Define model here
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(gene_counts.shape[-1],)))
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        model.add(tf.keras.layers.Dense(num_dense_nodes, activation=activation, name=name, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(24, activation=None))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics=[Metrics.CrossEntropy(), Metrics.Accuracy()],
                  weighted_metrics=[Metrics.CrossEntropy()])
    return model

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation):
    global best_evaluation
    global run_weights
    print('learning rate: ', learning_rate)
    print('weight_decay: ', weight_decay)
    print('dropout', dropout)
    print('num_dense_layers: ', num_dense_layers)
    print('num_dense_nodes: ', num_dense_nodes)
    print('activation: ', activation)
    print()
    model = create_model(learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout, num_dense_layers=num_dense_layers, num_dense_nodes=num_dense_nodes, activation=activation)

    model.fit(ds_train,
              steps_per_epoch=len(idx_train) // batch_size + 1,
              epochs=20000,
              validation_data=ds_valid,
              callbacks=callbacks)

    evaluation = model.evaluate(ds_valid)[-1]
    if evaluation < best_evaluation:
        run_weights.append(model.get_weights())
        best_evaluation = evaluation
    del model
    return evaluation


predictions = []
test_idx = []
parameters = []
model_weights = []
batch_size = 512
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat):
    test_idx.append(idx_test)
    run_weights = []
    best_evaluation = 100
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=batch_size, ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x: ((
                                          gene_loader(x),
                                           ),
                                          (
                                          y_label_loader(x),
                                          ),
                                           y_weights_loader(x)
                                          )
                            )

    ds_valid = tf.data.Dataset.from_tensor_slices((
                                                  (
                                                   gene_counts[idx_valid],
                                                   ),
                                                  (
                                                   y_label[idx_valid],
                                                  ),
                                                   y_weights[idx_valid]
                                                   ))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)

    ds_test = tf.data.Dataset.from_tensor_slices((
                                                 (
                                                  gene_counts[idx_test],
                                                 ),
                                                 (
                                                  y_label[idx_test],
                                                 ),

                                                  y_weights[idx_test]
                                                  ))

    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)

    search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=200, x0=default_paramaters, random_state=7, n_jobs=10)

    net = create_model(search_result.x[0],
                       search_result.x[1],
                       search_result.x[2],
                       search_result.x[3],
                       search_result.x[4],
                       search_result.x[5])
    net.set_weights(run_weights[-1])
    predictions.append(net.predict(ds_test))
    parameters.append(search_result.x)
    model_weights.append(run_weights[-1])


with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'gene_weights.pkl', 'wb') as f:
    pickle.dump([test_idx, model_weights, parameters], f)

P = np.concatenate(predictions)
#convert the model logits to probablities
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)

with open(cwd / 'figures' / 'tumor_classification' / 'project' / 'neural_net' / 'results' / 'gene_predictions.pkl', 'wb') as f:
    pickle.dump([test_idx, predictions], f)

print(np.sum((np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) * y_weights[np.concatenate(test_idx)]))
print(sum(np.argmax(predictions, axis=-1) == np.argmax(y_label[np.concatenate(test_idx)], axis=-1)) / len(y_label))
print(roc_auc_score(np.argmax(y_label[np.concatenate(test_idx)], axis=-1), predictions, multi_class='ovr'))

# 0.5257284609464159
# 0.5555555555555556
# 0.92897640353388
