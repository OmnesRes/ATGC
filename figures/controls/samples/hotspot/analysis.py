import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report
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


D, maf = pickle.load(open(cwd / 'figures' / 'controls' / 'data' / 'data.pkl', 'rb'))
sample_df = pickle.load(open(cwd / 'files' / 'tcga_sample_table.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

##bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])


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
genes_start = np.array([i + ':' + str(j) for i, j in zip(maf['Hugo_Symbol'].values, maf['Start_Position'].values)])
boolean = ['BRAF:140453136' in genes_start[j] for j in [np.where(D['sample_idx'] == i)[0] for i in range(sample_df.shape[0])]]
y_label = np.stack([[0, 1] if i else [1, 0] for i in boolean])
y_strat = np.argmax(y_label, axis=-1)
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)


with open(cwd / 'figures' / 'controls' / 'samples' / 'hotspot' / 'results' / 'weights_braf.pkl', 'rb') as f:
    weights = pickle.load(f)

sequence_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 16, 16])
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], output_dim=2, pooling='sum', mil_hidden=(64, 32, 16, 8), output_type='anlulogits')

test_idx = []
predictions = []
attentions = []
instance_labels = []
for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat)):
    mil.model.set_weights(weights[index])

    ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    ds_test = ds_test.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                           three_p_loader(x, ragged_output=True),
                                           ref_loader(x, ragged_output=True),
                                           alt_loader(x, ragged_output=True),
                                           strand_loader(x, ragged_output=True)),
                                           y,
                                          ))
    predictions.append(mil.model.predict(ds_test))
    test_idx.append(idx_test)
    attentions.append(mil.attention_model.predict(ds_test).to_list())
    instance_labels.append(genes_start[np.concatenate(np.array(indexes, dtype=object)[idx_test].tolist(), axis=1)[0]] == 'BRAF:140453136')




with open(cwd / 'figures' / 'controls' / 'samples' / 'hotspot' / 'results' / 'predictions_braf.pkl', 'wb') as f:
    pickle.dump([y_strat, test_idx, predictions], f)



print(classification_report(y_strat[np.concatenate(test_idx, axis=-1)], np.argmax(np.concatenate(predictions, axis=0), axis=-1), digits=4))


with open(cwd / 'figures' / 'controls' / 'samples' / 'hotspot' / 'results' / 'latent_braf.pkl', 'wb') as f:
    pickle.dump([attentions, instance_labels], f)

