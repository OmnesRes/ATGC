import numpy as np
import tensorflow as tf
from figures.msi.model.Sample_MIL import InstanceModels, RaggedModels
from figures.msi.model.KerasLayers import Losses, Metrics
from figures.msi.model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-4], True)
tf.config.experimental.set_visible_devices(physical_devices[-4], 'GPU')

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))


D, samples, sample_df = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[np.ones_like(D['strand'])]

indexes = [np.where(D['sample_idx'] == idx) for idx in range(sample_df.shape[0])]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')

dropout = .4
index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=dropout)
five_p_loader = DatasetsUtils.Map.FromNumpyandIndices(five_p, tf.int32)
three_p_loader = DatasetsUtils.Map.FromNumpyandIndices(three_p, tf.int32)
ref_loader = DatasetsUtils.Map.FromNumpyandIndices(ref, tf.int32)
alt_loader = DatasetsUtils.Map.FromNumpyandIndices(alt, tf.int32)
strand_loader = DatasetsUtils.Map.FromNumpyandIndices(strand, tf.float32)

five_p_loader_eval = DatasetsUtils.Map.FromNumpy(five_p, tf.int32)
three_p_loader_eval = DatasetsUtils.Map.FromNumpy(three_p, tf.int32)
ref_loader_eval = DatasetsUtils.Map.FromNumpy(ref, tf.int32)
alt_loader_eval = DatasetsUtils.Map.FromNumpy(alt, tf.int32)
strand_loader_eval = DatasetsUtils.Map.FromNumpy(strand, tf.float32)



# set y label and weights
y_label = samples['class']
cancer_labels = [i if i in ['STAD', 'UCEC', 'COAD'] else 'other' for i in samples['cancer']]
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(cancer_labels, y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(cancer_labels, y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)


weights = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_CE', min_delta=0.0001, patience=40, mode='min', restore_best_weights=True)]
losses = [Losses.CrossEntropy(from_logits=False)]

##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=9, random_state=0, shuffle=True).split(y_strat, y_strat):
    while True:
        idx_train_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]
        ds_train = tf.data.Dataset.from_tensor_slices((idx_train_train))
        ds_train = ds_train.apply(DatasetsUtils.Apply.SubSample(batch_size=128, ds_size=len(idx_train_train)))

        ds_train = ds_train.map(lambda x: ((index_loader(x), ) ), )

        ds_train = ds_train.map(lambda x: ((
                                               five_p_loader(x[0], x[1]),
                                               three_p_loader(x[0], x[1]),
                                               ref_loader(x[0], x[1]),
                                               alt_loader(x[0], x[1]),
                                               strand_loader(x[0], x[1])
                                           ),
                                           (
                                               y_label_loader(x[0]),
                                           ),
        ),
                                )
        ds_train.prefetch(1)
        ds_valid = tf.data.Dataset.from_tensor_slices(((
                                                           five_p_loader_eval(idx_valid),
                                                           three_p_loader_eval(idx_valid),
                                                           ref_loader_eval(idx_valid),
                                                           alt_loader_eval(idx_valid),
                                                           strand_loader_eval(idx_valid),
                                                       ),
                                                       (
                                                           y_label_loader(idx_valid),
                                                       ),
        ))
        ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)

        sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=64)
        mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], output_dims=[2], pooling='sum', mil_hidden=(64, 64, 32, 16), output_types=['classification_probability'], input_dropout=dropout)

        mil.model.compile(loss=losses,
                          metrics=[Metrics.CrossEntropy(from_logits=False), Metrics.Accuracy()],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                             clipvalue=10000))
        mil.model.fit(ds_train,
                      steps_per_epoch=10,
                      validation_data=ds_valid,
                      epochs=10000,
                      callbacks=callbacks)

        eval = mil.model.evaluate(ds_valid)
        if eval[1] < .06:
            break
        else:
            del mil
    weights.append(mil.model.get_weights())


with open(cwd / 'figures' / 'msi' / 'results' / 'run_dropout_ones_strand.pkl', 'wb') as f:
    pickle.dump(weights, f)