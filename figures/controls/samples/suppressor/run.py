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

features = [InputFeatures.VariantPositionBin(24, 100, {'position_loc': D['pos_loc'], 'position_bin': D['pos_bin'], 'chromosome': D['chr']})]

sample_features = ()

# set y label and weights
genes = maf['Hugo_Symbol'].values
boolean = ['PTEN' in genes[j] for j in [np.where(D['sample_idx'] == i)[0] for i in range(sample_df.shape[0])]]
y_label = np.stack([[0, 1] if i else [1, 0] for i in boolean])
y_strat = np.argmax(y_label, axis=-1)

# y_strat = np.ones(y_label.shape[0])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)

atgc = ATGC(features, sample_features=sample_features, aggregation_dimension=128, fusion_dimension=64)
atgc.build_instance_encoder_model(return_latent=False)
atgc.build_sample_encoder_model()
atgc.build_mil_model(output_dim=y_label.shape[1], output_extra=1, output_type='anlulogits', aggregation='recursion', mil_hidden=(16, 8))
metrics = [Losses.Weighted.CrossEntropyfromlogits.cross_entropy_weighted, Losses.Weighted.Accuracy.accuracy]
atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=10000), loss=Losses.Weighted.CrossEntropyfromlogits.cross_entropy_weighted, metrics=metrics)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_cross_entropy_weighted', min_delta=0.0001, patience=50, mode='min', restore_best_weights=True)]
initial_weights = atgc.mil_model.get_weights()

weights = []
##stratified K fold for test
for idx_train, idx_test in StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat):

    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    batch_gen_train = BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach='minibatch', batch_size=4500, idx_sample=idx_train)

    data_valid = next(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx_valid).data_generator())

    atgc.mil_model.set_weights(initial_weights)


    while True:
        atgc.mil_model.set_weights(initial_weights)
        atgc.mil_model.fit(batch_gen_train.data_generator(),
                           steps_per_epoch=batch_gen_train.n_splits*20,
                           epochs=10000,
                           validation_data=data_valid,
                           shuffle=False,
                           callbacks=callbacks)
        eval = atgc.mil_model.evaluate(data_valid[0], data_valid[1])
        if eval[2] > .99:
            break
        del atgc
        atgc = ATGC(features, sample_features=sample_features, aggregation_dimension=128, fusion_dimension=64)
        atgc.build_instance_encoder_model(return_latent=False)
        atgc.build_sample_encoder_model()
        atgc.build_mil_model(output_dim=y_label.shape[1], output_extra=1, output_type='anlulogits', aggregation='recursion', mil_hidden=(16, 8))
        atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=Losses.Weighted.CrossEntropyfromlogits.cross_entropy_weighted, metrics=metrics)
        initial_weights = atgc.mil_model.get_weights()

    weights.append(atgc.mil_model.get_weights())


with open(cwd / 'figures' / 'controls' / 'samples' / 'suppressor' / 'results' / 'weights.pkl', 'wb') as f:
    pickle.dump(weights, f)


