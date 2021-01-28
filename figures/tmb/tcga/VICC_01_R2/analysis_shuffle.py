import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
disable_eager_execution()
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

from model.CustomKerasModels import InputFeatures, ATGC
from model.CustomKerasTools import BatchGenerator, Losses

D, samples, maf, sample_df = pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'VICC_01_R2' / 'data' / 'data.pkl', 'rb'))
panels = pickle.load(open(cwd / 'files' / 'tcga_panel_table.pkl', 'rb'))

#shuffle the sample indexes
D['sample_idx'] = np.random.RandomState(seed=0).permutation(D['sample_idx'])

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

sample_features = ()

# set y label
y_label = np.log(sample_df['non_syn_counts'].values/(panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]
y_strat = np.argmax(samples['histology'], axis=-1)
y_label = np.repeat(y_label, 3, axis=-1)
metrics = [Losses.Weighted.QuantileLoss.quantile_loss]



all_features = [[InputFeatures.OnesLike({'position': D['pos_float'][:, np.newaxis]})],
            [InputFeatures.VariantPositionBin(
                24, 100, {'position_loc': D['pos_loc'], 'position_bin': D['pos_bin'], 'chromosome': D['chr']})],
            [InputFeatures.VariantSequence(6, 4, 2, [16, 16, 8, 8],
                                         {'5p': D['seq_5p'], '3p': D['seq_3p'], 'ref': D['seq_ref'], 'alt': D['seq_alt'], 'strand': D['strand_emb'], 'cds': D['cds_emb']},
                                         fusion_dimension=32,
                                         use_frame=False)]
            ]


all_weights = [pickle.load(open('figures/tmb/all_cancers/VICC_01_R2/results/run_naive_shuffle.pkl', 'rb')),
           pickle.load(open('figures/tmb/all_cancers/VICC_01_R2/results/run_position_shuffle.pkl', 'rb')),
           pickle.load(open('figures/tmb/all_cancers/VICC_01_R2/results/run_sequence_shuffle.pkl', 'rb'))
            ]

results = {}

for features, weights, name in zip(all_features, all_weights, ['naive', 'position', 'sequence']):

    atgc = ATGC(features, aggregation_dimension=64, fusion_dimension=32, sample_features=sample_features)
    atgc.build_instance_encoder_model(return_latent=False)
    atgc.build_sample_encoder_model()
    atgc.build_mil_model(output_dim=8, output_extra=1, output_type='quantiles', aggregation='recursion', mil_hidden=(16,))
    atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=Losses.Weighted.QuantileLoss.quantile_loss_weighted, metrics=metrics)

    ##test eval
    test_idx = []
    predictions = []
    genes = []
    evaluations = []

    for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat)):
        atgc.mil_model.set_weights(weights[index])
        data_test = next(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                         y_label=y_label, y_stratification=y_strat, sampling_approach=None, idx_sample=idx_test).data_generator())


        evaluations.append(atgc.mil_model.evaluate(data_test[0], data_test[1])[1])
        predictions.append(atgc.mil_model.predict(data_test[0])[0, :, :-1])
        test_idx.append(idx_test)

    #mse
    print(round(np.mean((y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])**2), 4))
    #mae
    print(round(np.mean(np.absolute(y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])), 4))
    #r2
    print(round(r2_score(np.concatenate(predictions)[:, 1], y_label[:, 0][np.concatenate(test_idx)]), 4))

    results[name] = np.concatenate(predictions)


results['y_true'] = y_label[np.concatenate(test_idx)]

##counting has to be nonsyn to nonsyn
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
panel_counts = maf[['Variant_Classification', 'Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x), (x['Variant_Classification'].isin(non_syn)).sum()], index=['panel_all_counts', 'panel_non_syn_counts']))
sample_df = pd.merge(sample_df, panel_counts, how='left', on='Tumor_Sample_Barcode')
sample_df.fillna({'panel_non_syn_counts': 0}, inplace=True)

results['counting'] = np.log(sample_df['panel_non_syn_counts'].values[np.concatenate(test_idx)] / (panels.loc[panels['Panel'] == 'VICC-01-R2']['cds'].values[0]/1e6) + 1)


##counting stats
#mse
print(round(np.mean((y_label[:, 0][np.concatenate(test_idx)] - results['counting'])**2), 4))
#mae
print(round(np.mean(np.absolute((y_label[:, 0][np.concatenate(test_idx)] - results['counting']))), 4))
#r2
print(round(r2_score(results['counting'], y_label[np.concatenate(test_idx)][:, 0]), 4))
