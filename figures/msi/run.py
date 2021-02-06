import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
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


D, samples, sample_df = pickle.load(open(cwd / 'figures' / 'msi' / 'data' / 'data.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

indexes = [np.where(D['sample_idx'] == idx) for idx in range(sample_df.shape[0])]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')

five_p_loader = DatasetsUtils.Map.FromNumpy(five_p, tf.int32)
three_p_loader = DatasetsUtils.Map.FromNumpy(three_p, tf.int32)
ref_loader = DatasetsUtils.Map.FromNumpy(ref, tf.int32)
alt_loader = DatasetsUtils.Map.FromNumpy(alt, tf.int32)
strand_loader = DatasetsUtils.Map.FromNumpy(strand, tf.float32)



# set y label and weights
y_label = samples['class']
cancer_labels = [i if i in ['STAD', 'UCEC', 'COAD'] else 'other' for i in samples['cancer']]
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(cancer_labels, y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(cancer_labels, y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)


weights = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_CE', min_delta=0.0001, patience=50, mode='min', restore_best_weights=True)]
losses = [Losses.CrossEntropy(from_logits=False)]
sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8])
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], output_dim=2, pooling='sum', mil_hidden=(64, 64, 32, 16), output_type='classification_probability')
mil.model.compile(loss=losses,
                  metrics=[Metrics.CrossEntropy(from_logits=False), Metrics.Accuracy()],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                     clipvalue=10000))
initial_weights = mil.model.get_weights()

##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=9, random_state=0, shuffle=True).split(y_strat, y_strat):
    ##due to the y_strat levels not being constant this idx_train/idx_valid split is not deterministic
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.SubSample(batch_size=128, ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                           three_p_loader(x, ragged_output=True),
                                           ref_loader(x, ragged_output=True),
                                           alt_loader(x, ragged_output=True),
                                           strand_loader(x, ragged_output=True)),
                                           y
                                           ))

    ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid]))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
    ds_valid = ds_valid.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                           three_p_loader(x, ragged_output=True),
                                           ref_loader(x, ragged_output=True),
                                           alt_loader(x, ragged_output=True),
                                           strand_loader(x, ragged_output=True)),
                                           y
                                           ))

    while True:
        mil.model.set_weights(initial_weights)
        mil.model.fit(ds_train,
                      steps_per_epoch=10,
                      validation_data=ds_valid,
                      epochs=10000,
                      callbacks=callbacks,
                      workers=4)

        eval = mil.model.evaluate(ds_valid)
        if eval[1] < .07:
            break
        else:
            sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8])
            mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], output_dim=2, pooling='sum', mil_hidden=(64, 64, 32, 16), output_type='classification_probability')
            mil.model.compile(loss=losses,
                              metrics=[Metrics.CrossEntropy(from_logits=False), Metrics.Accuracy()],
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                                 clipvalue=10000))
            initial_weights = mil.model.get_weights()
    weights.append(mil.model.get_weights())



with open(cwd / 'figures' / 'msi' / 'results' / 'run.pkl', 'wb') as f:
    pickle.dump(weights, f)


