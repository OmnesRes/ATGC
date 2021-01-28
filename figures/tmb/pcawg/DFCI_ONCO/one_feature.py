import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
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

D, samples, maf, sample_df = pickle.load(open(cwd / 'figures' / 'tmb' / 'pcawg' / 'DFCI_ONCO' / 'data' / 'data.pkl', 'rb'))
panels = pickle.load(open(cwd / 'files' / 'pcawg_panel_table.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand'].astype(int)]

chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]

frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds'].astype(int)]

hist_emb_mat = np.concatenate([np.zeros(samples['histology'].shape[1])[np.newaxis, :], np.diag(np.ones(samples['histology'].shape[1]))], axis=0)
samples['hist_emb'] = hist_emb_mat[np.argmax(samples['histology'], axis=-1)]

##bin position
def pos_one_hot(pos):
    one_pos = int(pos * 100)
    return one_pos, (pos * 100) - one_pos

result = np.apply_along_axis(pos_one_hot, -1, D['pos_float'][:, np.newaxis])

D['pos_bin'] = np.stack(result[:, 0]) + 1
D['pos_loc'] = np.stack(result[:, 1])


#
#
# features = [InputFeatures.VariantSequence(6, 4, 2, [16, 16, 8, 8],
#                                          {'5p': D['seq_5p'], '3p': D['seq_3p'], 'ref': D['seq_ref'], 'alt': D['seq_alt'], 'strand': D['strand_emb'], 'cds': D['cds_emb']},
#                                          fusion_dimension=32,
#                                          use_frame=False)]
#
# features = [InputFeatures.VariantPositionBin(24, 100, {'position_loc': D['pos_loc'], 'position_bin': D['pos_bin'], 'chromosome': D['chr']})]
#
features = [InputFeatures.OnesLike({'position': D['pos_float'][:, np.newaxis]})]


sample_features = ()

# set y label
y_label = np.log(sample_df['non_syn_counts'].values / (panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0]/1e6) + 1)[:, np.newaxis]
y_strat = np.argmax(samples['histology'], axis=-1)
y_label = np.repeat(y_label, 3, axis=-1)

runs = 3
initial_weights = []
metrics = [Losses.Weighted.QuantileLoss.quantile_loss]
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=40, mode='min', restore_best_weights=True)]

for i in range(runs):
    atgc = ATGC(features, aggregation_dimension=64, fusion_dimension=32, sample_features=sample_features)
    atgc.build_instance_encoder_model(return_latent=False)
    atgc.build_sample_encoder_model()
    atgc.build_mil_model(output_dim=8, output_extra=1, output_type='quantiles', aggregation='recursion', mil_hidden=(16,))
    atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=Losses.Weighted.QuantileLoss.quantile_loss, metrics=metrics)
    initial_weights.append(atgc.mil_model.get_weights())

weights = []
##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat):

    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    batch_gen_train = BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, sampling_approach='minibatch', batch_size=128, idx_sample=idx_train)

    data_valid = next(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, sampling_approach=None, idx_sample=idx_valid).data_generator())

    eval = 100
    for initial_weight in initial_weights:
        atgc.mil_model.set_weights(initial_weight)
        atgc.mil_model.fit(batch_gen_train.data_generator(),
                                   steps_per_epoch=batch_gen_train.n_splits*2,
                                   epochs=10000,
                                   validation_data=data_valid,
                                   shuffle=False,
                                   callbacks=callbacks)
        run_eval = atgc.mil_model.evaluate(data_valid[0], data_valid[1])[1]
        if run_eval < eval:
            eval = run_eval
            run_weights = atgc.mil_model.get_weights()

    weights.append(run_weights)


with open(cwd / 'figures' / 'tmb' / 'pcawg' / 'DFCI_ONCO' / 'results' / 'run_naive.pkl', 'wb') as f:
    pickle.dump(weights, f)


