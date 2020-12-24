from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sim_data.sim_data_tools import *
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sim_data.pkl', 'rb'))

instance_sum_evaluations, instance_sum_histories, weights = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'instance_model_sum.pkl', 'rb'))
# sample_sum_evaluations, sample_sum_histories, weights = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_sum.pkl', 'rb'))

import tensorflow as tf
from model.Instance_MIL import InstanceModels, RaggedModels
# from model.Sample_MIL import InstanceModels, RaggedModels
from model import DatasetsUtils
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[4], True)
tf.config.experimental.set_visible_devices(physical_devices[4], 'GPU')

##perform embeddings with a zero vector for index 0
strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

indexes = [np.where(D['sample_idx'] == idx) for idx in range(len(samples['classes']))]

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

cancer_strat = np.zeros_like(samples['classes']) ##no cancer info in this simulated data
y_label = np.stack(np.concatenate([samples['times'][:, np.newaxis], samples['event'][:, np.newaxis], cancer_strat[:, np.newaxis]], axis=-1))
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(samples['classes'], y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(samples['classes'], y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))


ds_all = tf.data.Dataset.from_tensor_slices((np.arange(len(y_label)), y_label))
ds_all = ds_all.batch(len(y_label), drop_remainder=False)
ds_all = ds_all.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                       three_p_loader(x, ragged_output=True),
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                       strand_loader(x, ragged_output=True)),
                                       y))



cancer_test_ranks = {}
cancer_test_indexes = {}
cancer_test_expectation_ranks = {}

for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat)):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]
    tile_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])
    mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model], output_dim=1, pooling='sum', output_type='other')
    mil.model.set_weights(weights[index])
    y_pred_all = mil.model.predict(ds_all)
    ##get ranks per cancer
    for index, cancer in enumerate(['NA']):
        mask = np.where(cancer_strat == index)[0]
        cancer_test_indexes[cancer] = cancer_test_indexes.get(cancer, []) + [mask[np.isin(mask, idx_test, assume_unique=True)]]
        temp = np.exp(-y_pred_all[mask, 0]).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(mask))
        cancer_test_ranks[cancer] = cancer_test_ranks.get(cancer, []) + [ranks[np.isin(mask, idx_test, assume_unique=True)]]

indexes = np.concatenate(cancer_test_indexes['NA'])
ranks = np.concatenate(cancer_test_ranks['NA'])
concordance_index(samples['times'][indexes], ranks, samples['event'][indexes])


sample_df = pd.DataFrame(data={'class': samples['classes'][idx_test],
                               'predictions': y_pred_all[:, 0][idx_test],
                               })

with open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'instance_model_sum_eval.pkl', 'wb') as f:
    pickle.dump([indexes, ranks, sample_df], f)

