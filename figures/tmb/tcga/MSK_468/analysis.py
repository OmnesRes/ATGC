import numpy as np
import tensorflow as tf
import pandas as pd
from model.Sample_MIL import InstanceModels, RaggedModels
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

pos_loc = np.array([D['pos_loc'][i] for i in indexes], dtype='object')
pos_bin = np.array([D['pos_bin'][i] for i in indexes], dtype='object')
chr = np.array([D['chr'][i] for i in indexes], dtype='object')

pos_loader = DatasetsUtils.Map.FromNumpy(pos_loc, tf.float32)
bin_loader = DatasetsUtils.Map.FromNumpy(pos_bin, tf.float32)
chr_loader = DatasetsUtils.Map.FromNumpy(chr, tf.int32)

ones_loader = DatasetsUtils.Map.FromNumpy(np.array([np.ones_like(D['pos_loc'])[i] for i in indexes], dtype='object'), tf.float32)


loaders = [
    [ones_loader],
    [pos_loader, bin_loader, chr_loader],
    [five_p_loader, three_p_loader, ref_loader, alt_loader, strand_loader],
]


# set y label
y_label = np.log(sample_df['non_syn_counts'].values/(panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]
y_strat = np.argmax(samples['histology'], axis=-1)

losses = [Losses.QuantileLoss()]
metrics = [Metrics.QuantileLoss()]

encoders = [InstanceModels.PassThrough(shape=(1,)),
            InstanceModels.VariantPositionBin(24, 100),
            InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8], fusion_dimension=32)]

all_weights = [
    pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_468' / 'results' / 'run_naive.pkl', 'rb')),
    pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_468' / 'results' / 'run_position.pkl', 'rb')),
    pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_468' / 'results' / 'run_sequence.pkl', 'rb'))
    ]

results = {}

for encoder, loaders, weights, name in zip(encoders, loaders, all_weights, ['naive', 'position', 'sequence']):

    mil = RaggedModels.MIL(instance_encoders=[encoder.model], output_dim=1, pooling='sum', mil_hidden=(64, 32, 16), output_type='quantiles', regularization=0)
    mil.model.compile(loss=losses,
                      metrics=metrics,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    ##test eval
    test_idx = []
    predictions = []
    genes = []
    evaluations = []

    for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat)):
        mil.model.set_weights(weights[index])

        ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
        ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
        ds_test = ds_test.map(lambda x, y: (tuple([i(x, ragged_output=True) for i in loaders]),
                                            y,
                                            ))

        evaluations.append(mil.model.evaluate(ds_test)[1])
        predictions.append(mil.model.predict(ds_test))
        test_idx.append(idx_test)


    #mse
    print(round(np.mean((y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])**2), 4))
    #mae
    print(round(np.mean(np.absolute(y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])), 4))
    #r2
    print(round(r2_score(y_label[:, 0][np.concatenate(test_idx)], np.concatenate(predictions)[:, 1]), 4))

    results[name] = np.concatenate(predictions)


results['y_true'] = y_label[np.concatenate(test_idx)]

##counting has to be nonsyn to nonsyn
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
panel_counts = maf[['Variant_Classification', 'Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x), (x['Variant_Classification'].isin(non_syn)).sum()], index=['panel_all_counts', 'panel_non_syn_counts']))
sample_df = pd.merge(sample_df, panel_counts, how='left', on='Tumor_Sample_Barcode')
sample_df.fillna({'panel_non_syn_counts': 0}, inplace=True)

results['counting'] = np.log(sample_df['panel_non_syn_counts'].values[np.concatenate(test_idx)] / (panels.loc[panels['Panel'] == 'MSK-IMPACT468']['cds'].values[0]/1e6) + 1)

with open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_468' / 'results' / 'predictions.pkl', 'wb') as f:
    pickle.dump(results, f)



##counting stats
#mse
print(round(np.mean((y_label[:, 0][np.concatenate(test_idx)] - results['counting'])**2), 4))
#mae
print(round(np.mean(np.absolute((y_label[:, 0][np.concatenate(test_idx)] - results['counting']))), 4))
#r2
print(round(r2_score(y_label[np.concatenate(test_idx)][:, 0], results['counting']), 4))
