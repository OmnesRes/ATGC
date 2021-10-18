import numpy as np
import tensorflow as tf
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

##load the instance and sample data
D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_1' / 'sim_data.pkl', 'rb'))

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

y_label = np.log(np.array(samples['values']) + 1)[:, np.newaxis]
y_strat = np.ones_like(y_label)

idx_train, idx_test = next(StratifiedShuffleSplit(random_state=0, n_splits=1, test_size=200).split(y_strat, y_strat))
idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
ds_test = ds_test.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                       three_p_loader(x, ragged_output=True),
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                       strand_loader(x, ragged_output=True)),
                                       ))

from model.Sample_MIL import InstanceModels, RaggedModels
evaluations, histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_1' / 'sample_model_mean.pkl', 'rb'))

predictions = []
attentions = []
for i in range(3):
    tile_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])
    # mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model], output_dims=[1], pooling='both', output_types=['regression'], pooled_layers=[32, ])
    mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model], output_dims=[1], pooling='mean', output_types=['regression'], mode='none')
    mil.model.set_weights(weights[i])
    predictions.append(mil.model.predict(ds_test))
    # attentions.append(mil.attention_model.predict(ds_test).to_list())
#
with open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_1' / 'sample_model_mean_predictions.pkl', 'wb') as f:
    pickle.dump([idx_test, predictions], f)