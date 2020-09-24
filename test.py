import numpy as np
import pandas as pd
import tensorflow as tf
from KerasModels import InstanceModels, RaggedModels, SampleModels
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[-1], True)
# tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')


D, samples, maf, sample_df = pickle.load(open('/home/jordan/Desktop/ATGC/figures/tmb/tcga/MSK_468/data/data.pkl', 'rb'))
panels = pickle.load(open('/home/jordan/Desktop/ATGC/files/tcga_panel_table.pkl', 'rb'))


mask = ~pd.isna(sample_df['OS.time']) & ~pd.isna(sample_df['age_at_initial_pathologic_diagnosis'])

sample_df = sample_df.loc[mask]

samples['histology'] = samples['histology'][mask]

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]


#bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])

# indexes = np.argsort(D['sample_idx'])
# ones = tf.RaggedTensor.from_value_rowids(np.ones_like(D['pos_loc'])[indexes].astype(np.float32), D['sample_idx'][indexes], nrows=sample_df.shape[0])

y_strat = np.argmax(samples['histology'], axis=-1)
y_label = np.stack(np.concatenate([sample_df[['OS.time', 'OS']].to_records(index=False, column_dtypes='float64').tolist(), y_strat[:, np.newaxis]], axis=-1))

age = sample_df.age_at_initial_pathologic_diagnosis.values[:, np.newaxis]

tfds_train = tf.data.Dataset.from_tensor_slices(((samples['histology'], age), y_label))
tfds_train = tfds_train.shuffle(len(y_label), reshuffle_each_iteration=True).batch(len(y_label), drop_remainder=True)

tile_encoder = InstanceModels.PassThrough(shape=(1,))
sample_encoder_group = SampleModels.PassThrough(shape=samples['histology'].shape[1:])
sample_encoder_age = SampleModels.PassThrough(shape=age.shape[1:])



mil = RaggedModels.MIL(instance_encoders=[], sample_encoders=[sample_encoder_group.model, sample_encoder_age.model], output_dim=1, output_type='survival')
losses = [RaggedModels.losses.CoxPH()]
mil.compile(loss=losses)


mil.fit(tfds_train, epochs=200)

tfds_all = tf.data.Dataset.from_tensor_slices(((samples['histology'], age), y_label))
tfds_all = tfds_all.batch(len(y_label), drop_remainder=False)
results = mil.predict(tfds_all)

concordances = []
for i in range(max(y_strat) +1):
    mask = np.where(y_strat == i)
    temp = np.exp(-results[:, 0][mask]).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(results[:, 0][mask]))
    concordance = concordance_index(sample_df['OS.time'].values[mask], ranks, sample_df['OS'].values[mask])
    concordances.append(concordance)

#
#
# temp = np.exp(-results[:, 0]).argsort()
# ranks = np.empty_like(temp)
# ranks[temp] = np.arange(len(results))
#
# print(concordance_index(sample_df['OS.time'].values, ranks, sample_df['OS'].values))