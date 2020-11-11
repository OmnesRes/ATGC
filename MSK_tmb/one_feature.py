import numpy as np
import tensorflow as tf
from model.Sample_MIL import InstanceModels, RaggedModels, SampleModels
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

D, samples, maf, sample_df = pickle.load(open(cwd / 'MSK_tmb' / 'data' / 'all_cancers_data.pkl', 'rb'))
panels = pickle.load(open(cwd / '..' / 'ATGC_paper' / 'files' / 'tcga_panel_table.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

indexes = [np.where(D['sample_idx'] == idx) for idx in range(len(samples['histology']))]

five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')

##bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])
ones = np.array([np.ones_like(D['pos_loc'][i]) for i in indexes], dtype='object')

# set y label
y_label = np.log(sample_df['non_syn_counts'].values / (panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]
y_strat = np.argmax(samples['histology'], axis=-1)

runs = 1
initial_weights = []
metrics = [RaggedModels.losses.QuantileLoss()]
losses = [RaggedModels.losses.QuantileLoss()]
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_quantile_loss', min_delta=0.0001, patience=40, mode='min', restore_best_weights=True)]

for i in range(runs):
    # tile_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])
    tile_encoder = InstanceModels.PassThrough(shape=(1, ))
    sample_encoder = SampleModels.PassThrough(shape=(samples['histology'].shape[-1], ))
    mil = RaggedModels.MIL(mode='aggregation', instance_encoders=[tile_encoder.model], pooled_layers=[64], sample_layers=[64, 32], sample_encoders=[sample_encoder.model], output_dim=1, output_type='quantiles')
    mil.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses, metrics=metrics)
    initial_weights.append(mil.model.get_weights())

weights = []
##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat):

    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1500, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train], y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=512, ds_size=len(idx_train)))
    # x_loader = DatasetsUtils.Map.LoadBatchIndex(loaders=[DatasetsUtils.Loaders.FromNumpy(five_p, tf.int32),
    #                                                      DatasetsUtils.Loaders.FromNumpy(three_p, tf.int32),
    #                                                      DatasetsUtils.Loaders.FromNumpy(ref, tf.int32),
    #                                                      DatasetsUtils.Loaders.FromNumpy(alt, tf.int32),
    #                                                      DatasetsUtils.Loaders.FromNumpy(strand, tf.float32)])

    x_loader = DatasetsUtils.Map.LoadBatchIndex(loaders=[DatasetsUtils.Loaders.FromNumpy(ones, tf.float32),
                                                         DatasetsUtils.Loaders.FromNumpy(np.array([[i] for i in samples['histology']], dtype='object'), tf.float32)])

    # ds_train = ds_train.map(lambda x, y: (x_loader(x, to_ragged=[True, True, True, True, True]), y))
    ds_train = ds_train.map(lambda x, y: (x_loader(x, to_ragged=[True, False]), y))

    ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid]))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
    # ds_valid = ds_valid.map(lambda x, y: (x_loader(x, to_ragged=[True, True, True, True, True]), y))
    ds_valid = ds_valid.map(lambda x, y: (x_loader(x, to_ragged=[True, False]), y))
    eval = 100
    for initial_weight in initial_weights:
        mil.model.set_weights(initial_weight)
        mil.model.fit(ds_train, validation_data=ds_valid,
                      epochs=10000,
                      callbacks=callbacks)
        run_eval = mil.model.evaluate(ds_valid)[1]
        if run_eval < eval:
            eval = run_eval
            run_weights = mil.model.get_weights()

    weights.append(run_weights)


# with open('tmb_weights.pkl', 'wb') as f:
#     pickle.dump(weights, f)



##test eval
test_idx = []
predictions = []


for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat)):
    mil.model.set_weights(weights[index])
    ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    ds_test = ds_test.map(lambda x, y: (x_loader(x, to_ragged=[True, False]), y))
    predictions.append(mil.model.predict(ds_test))
    test_idx.append(idx_test)
from sklearn.metrics import r2_score

#mse
print(round(np.mean((y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])**2), 4))
#mae
print(round(np.mean(np.absolute(y_label[:, 0][np.concatenate(test_idx)] - np.concatenate(predictions)[:, 1])), 4))
#r2
print(round(r2_score(np.concatenate(predictions)[:, 1], y_label[:, 0][np.concatenate(test_idx)]), 4))

#
#
# import pandas as pd
# ##counting has to be nonsyn to nonsyn
# panel_counts = maf[['Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x)], index=['panel_all_counts']))
# sample_df = pd.merge(sample_df, panel_counts, how='left', on='Tumor_Sample_Barcode')
# sample_df.fillna({'panel_all_counts': 0}, inplace=True)
#
# results={}
# results['counting'] = np.log(sample_df['panel_all_counts'].values[np.concatenate(test_idx)] / (panels.loc[panels['Panel'] == 'MSK-IMPACT468']['cds'].values[0]/1e6) + 1)
#
#
# ##counting stats
# #mse
# print(round(np.mean((y_label[:, 0][np.concatenate(test_idx)] - results['counting'])**2), 4))
# #mae
# print(round(np.mean(np.absolute((y_label[:, 0][np.concatenate(test_idx)] - results['counting']))), 4))
# #r2
# print(round(r2_score(results['counting'], y_label[np.concatenate(test_idx)][:, 0]), 4))
