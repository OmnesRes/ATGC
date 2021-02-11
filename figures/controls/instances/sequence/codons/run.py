import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from sklearn.model_selection import StratifiedShuffleSplit
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

categories = ['Missense_Mutation', 'Silent',\
        "3'UTR", 'Nonsense_Mutation', 'Intron', 'Frame_Shift_Del', "5'UTR", 'Splice_Site', 'Frame_Shift_Ins',\
        'In_Frame_Del', 'In_Frame_Ins']

boolean = maf['Variant_Classification'].isin(categories)
maf = maf.loc[boolean]
maf.reset_index(inplace=True)
for i in D:
    D[i] = D[i][boolean]

##group the noncoding labels
label_dict = {i: i for i in categories}
label_dict["3'UTR"] = 'noncoding'
label_dict["5'UTR"] = 'noncoding'
label_dict["Intron"] = 'noncoding'
maf['Variant_Classification'] = maf.Variant_Classification.apply(lambda x: label_dict[x])

A = maf.Variant_Classification.astype('category')
variant_labels = A.cat.categories.values
D['classification'] = np.eye(len(variant_labels))[A.cat.codes]

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

# set y label and weights
y_strat = np.arange(9)[np.argwhere(D['classification'])[:, 1]]
y_label = D['classification']
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

# set train, valid, test indices - with stratification if needed
idx_train, idx_test = list(StratifiedShuffleSplit(n_splits=1, test_size=200000, random_state=0).split(y_strat, y_strat))[0]
idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

batch_size = 50000
ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train], y_weights[idx_train]))
ds_train = ds_train.shuffle(len(idx_train), reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True).repeat()
ds_train = ds_train.map(lambda x, y, z: ((tf.gather(tf.constant(D['seq_5p'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_3p'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_ref'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_alt'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['strand_emb'], dtype=tf.float32), x),
                                      tf.gather(tf.constant(D['cds_emb'], dtype=tf.float32), x)
                                       ),
                                       y,
                                       z
                                       ))

ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid], y_weights[idx_valid]))
ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
ds_valid = ds_valid.map(lambda x, y, z: ((tf.gather(tf.constant(D['seq_5p'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_3p'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_ref'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_alt'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['strand_emb'], dtype=tf.float32), x),
                                      tf.gather(tf.constant(D['cds_emb'], dtype=tf.float32), x)
                                       ),
                                       y,
                                       z
                                       ))




ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
ds_test = ds_test.map(lambda x, y: ((tf.gather(tf.constant(D['seq_5p'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_3p'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_ref'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['seq_alt'], dtype=tf.int32), x),
                                      tf.gather(tf.constant(D['strand_emb'], dtype=tf.float32), x),
                                      tf.gather(tf.constant(D['cds_emb'], dtype=tf.float32), x)
                                       ),
                                       y,
                                      ))



sequence_encoder = InstanceModels.VariantSequence(6, 4, 2, [64, 64, 64, 64], fusion_dimension=128, use_frame=True)
mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[sequence_encoder.model], output_dim=y_label.shape[-1], output_type='other', mil_hidden=[128, 128, 64, 32], mode='none')
losses = [Losses.CrossEntropy()]
mil.model.compile(loss=losses,
                  metrics=[Metrics.Accuracy(), Metrics.CrossEntropy()],
                  weighted_metrics=[Metrics.Accuracy(), Metrics.CrossEntropy()],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                     ))

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_weighted_CE', min_delta=0.001, patience=10, mode='min', restore_best_weights=True)]


mil.model.fit(ds_train, steps_per_epoch=50,
              validation_data=ds_valid,
              epochs=10000,
              callbacks=callbacks,
              )


with open(cwd / 'figures' / 'controls' / 'instances' / 'sequence' / 'codons' / 'results' / 'weights_with_frame.pkl', 'wb') as f:
    pickle.dump(mil.model.get_weights(), f)

P = mil.model.predict(ds_test)
z = np.exp(P - np.max(P, axis=1, keepdims=True))
predictions = z / np.sum(z, axis=1, keepdims=True)

y_true = np.argmax(y_label[idx_test], axis=-1)
y_pred = np.argmax(predictions, axis=-1)

matrix = confusion_matrix(y_true, y_pred)

with open(cwd / 'figures' / 'controls' / 'instances' / 'sequence' / 'codons' / 'results' / 'matrix_with_frame.pkl', 'wb') as f:
    pickle.dump(matrix, f)

