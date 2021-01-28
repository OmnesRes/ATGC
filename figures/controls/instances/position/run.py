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

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

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

maf['label'] = np.zeros(maf.shape[0])
maf['label'][maf['Hugo_Symbol'] == 'MLH1'] = 1
maf['label'][maf['Hugo_Symbol'] == 'MSH2'] = 2
maf['label'][maf['Hugo_Symbol'] == 'MSH6'] = 3
maf['label'][maf['Hugo_Symbol'] == 'PMS2'] = 4
maf['label'][maf['Hugo_Symbol'] == 'TP53'] = 5
maf['label'][maf['Hugo_Symbol'] == 'PTEN'] = 6
maf['label'][maf['Hugo_Symbol'] == 'KRAS'] = 7

A = maf['label'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]

y_label = classes_onehot
y_strat = np.argmax(y_label, axis=-1)

class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))
y_weights = np.array([1 / class_counts[_] for _ in y_strat])
y_weights /= np.sum(y_weights)


atgc = ATGC(features, sample_features=sample_features, fusion_dimension=32)
atgc.build_instance_encoder_model(return_latent=False)
atgc.build_sample_encoder_model()
atgc.build_mil_model(output_dim=y_label.shape[1], output_extra=1, output_type='anlulogits', aggregation='none', mil_hidden=(16,))
metrics = [Losses.Weighted.CrossEntropyfromlogits.cross_entropy, Losses.Weighted.Accuracy.accuracy]
atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=Losses.Weighted.CrossEntropyfromlogits.cross_entropy, metrics=metrics)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_cross_entropy', min_delta=0.00001, patience=100, mode='min', restore_best_weights=True)]
initial_weights = atgc.mil_model.get_weights()

weights = []
test_idx = []


for idx_train, idx_test in StratifiedKFold(n_splits=2, random_state=0).split(y_strat, y_strat):

    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=500000, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    batch_gen_train = BatchGenerator(x_instance_sample_idx=None, x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach='subsample', batch_size=len(idx_train), idx_sample=idx_train)


    data_valid = next(BatchGenerator(x_instance_sample_idx=None, x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx_valid).data_generator())

    data_train = next(batch_gen_train.data_generator())

    while True:
        atgc.mil_model.set_weights(initial_weights)

        atgc.mil_model.fit(x=data_train[0],
                           y=data_train[1],
                           batch_size=1,
                           epochs=10000,
                           validation_data=data_valid,
                           shuffle=False,
                           callbacks=callbacks)

        eval = atgc.mil_model.evaluate(data_valid[0], data_valid[1])
        if eval[1] < .0005:
            break
        del atgc
        atgc = ATGC(features, sample_features=sample_features, fusion_dimension=32)
        atgc.build_instance_encoder_model(return_latent=False)
        atgc.build_sample_encoder_model()
        atgc.build_mil_model(output_dim=y_label.shape[1], output_extra=1, output_type='anlulogits', aggregation='none', mil_hidden=(16,))
        atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=Losses.Weighted.CrossEntropyfromlogits.cross_entropy, metrics=metrics)
        initial_weights = atgc.mil_model.get_weights()
    weights.append(atgc.mil_model.get_weights())
    test_idx.append(idx_test)


with open('figures/controls/instances/position/results/weights.pkl', 'wb') as f:
    pickle.dump(weights, f)