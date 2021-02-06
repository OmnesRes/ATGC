import numpy as np
import tensorflow as tf
import pandas as pd
from model.Sample_MIL import InstanceModels, RaggedModels, SampleModels
from model.KerasLayers import Losses, Metrics
from model import DatasetsUtils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
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


D, samples, maf, sample_df = pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_468' / 'data' / 'data.pkl', 'rb'))
panels = pickle.load(open(cwd / 'files' / 'tcga_panel_table.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

hist_emb_mat = np.concatenate([np.zeros(samples['histology'].shape[1])[np.newaxis, :], np.diag(np.ones(samples['histology'].shape[1]))], axis=0)
samples['hist_emb'] = hist_emb_mat[np.argmax(samples['histology'], axis=-1)]

counts = np.array([len(np.where(D['sample_idx'] == idx)[0]) for idx in range(sample_df.shape[0])])[:, np.newaxis]

# set y label
y_label = np.log(sample_df['non_syn_counts'].values/(panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]


##if using tcga cancer types
y_strat = np.argmax(samples['histology'], axis=-1)

##if using NCI-T labels
# label_counts = sample_df['NCI-T Label'].value_counts().to_dict()
# mask = sample_df['NCI-T Label'].map(lambda x: label_counts.get(x, 0) >= 36)
# y_label = y_label[mask]
# counts = counts[mask]
# labels = [i for i in sorted(label_counts.keys()) if label_counts[i] >= 36]
# y_strat = sample_df['NCI-T Label'][mask].map(lambda x: labels.index(x)).values


losses = [Losses.QuantileLoss()]
metrics = [Metrics.QuantileLoss()]

pass_encoder = InstanceModels.PassThrough(shape=(1,))
type_encoder = SampleModels.Type(shape=(), dim=max(y_strat) + 1)

weights = pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_468' / 'results' / 'run_naive_sample_tcga.pkl', 'rb'))

mil = RaggedModels.MIL(sample_encoders=[pass_encoder.model, type_encoder.model], output_dim=1, mil_hidden=(64, 32, 16), output_type='quantiles', regularization=0, mode='none')

##test eval
test_idx = []
predictions = []

for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat)):
    mil.model.set_weights(weights[index])

    ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    ds_test = ds_test.map(lambda x, y: ((
                                           tf.gather(tf.constant(counts), x),
                                           # tf.gather(tf.constant(y_strat + 1), x),
                                           tf.gather(tf.constant(np.zeros_like(y_strat)), x),

                                            ),
                                           y,
                                           ))

    predictions.append(mil.model.predict(ds_test))
    test_idx.append(idx_test)


#mse
print(round(np.mean((y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])**2), 4))
#mae
print(round(np.mean(np.absolute(y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])), 4))
#r2
print(round(r2_score(y_label[:, 0][np.concatenate(test_idx)], np.concatenate(predictions)[:, 1]), 4))


