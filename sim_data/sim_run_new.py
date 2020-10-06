import numpy as np
import pandas as pd
import tensorflow as tf
from model_new.KerasModels import InstanceModels, RaggedModels, SampleModels
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

##load the instance and sample data
D, samples = pickle.load(open(cwd / 'sim_data' / 'sim_data.pkl', 'rb'))

##perform embeddings with a zero vector for index 0
strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

indexes = np.argsort(D['sample_idx'])

five_p = tf.RaggedTensor.from_value_rowids(D['seq_5p'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
three_p = tf.RaggedTensor.from_value_rowids(D['seq_3p'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
ref = tf.RaggedTensor.from_value_rowids(D['seq_ref'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
alt = tf.RaggedTensor.from_value_rowids(D['seq_alt'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
strand = tf.RaggedTensor.from_value_rowids(D['strand_emb'][indexes].astype(np.float32), D['sample_idx'][indexes], nrows=len(samples['classes']))

y_label = np.stack([[0, 1] if i == 1 else [1, 0] for i in samples['classes']])
y_strat = np.argmax(y_label, axis=-1)

idx_train, idx_test = next(StratifiedShuffleSplit(n_splits=1, test_size=100).split(y_strat, y_strat))

train_data = (tf.gather(five_p, idx_train), tf.gather(three_p, idx_train), tf.gather(ref, idx_train), tf.gather(alt, idx_train), tf.gather(strand, idx_train))
valid_data = (tf.gather(five_p, idx_test), tf.gather(three_p, idx_test), tf.gather(ref, idx_test), tf.gather(alt, idx_test), tf.gather(strand, idx_test))

tfds_train = tf.data.Dataset.from_tensor_slices((train_data, y_label[idx_train]))
tfds_train = tfds_train.shuffle(len(y_label), reshuffle_each_iteration=True).batch(len(idx_train), drop_remainder=True)

tfds_valid = tf.data.Dataset.from_tensor_slices((valid_data, y_label[idx_test]))
tfds_valid = tfds_valid.batch(len(idx_test), drop_remainder=False)

tile_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])

mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model], sample_encoders=[], output_dim=2, attention_heads=3, output_type='other')

losses = [tf.keras.losses.CategoricalCrossentropy(from_logits=True)]
mil.model.compile(loss=losses, metrics='accuracy')

mil.model.fit(tfds_train, validation_data=tfds_valid, epochs=1000)

attention = mil.attention_model.predict(tfds_valid)

attention_attention = [k for i in attention[1].to_list() for j in i for k in j]
# attention = [k for i in attention for j in i for k in j]


# attention_class_1 = [j[0] for i in attention for j in i]
# attention_class_2 = [j[1] for i in attention for j in i]
#
import pylab as plt
plt.hist(attention_attention, bins=100)
plt.show()