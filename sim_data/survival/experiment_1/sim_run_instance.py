import numpy as np
import tensorflow as tf
from model.Instance_MIL import InstanceModels, RaggedModels
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from lifelines.utils import concordance_index
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[4], True)
tf.config.experimental.set_visible_devices(physical_devices[4], 'GPU')
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

##load the instance and sample data
D, samples = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sim_data.pkl', 'rb'))

##perform embeddings with a zero vector for index 0
strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]

indexes = np.argsort(D['sample_idx'])

five_p = tf.RaggedTensor.from_value_rowids(D['seq_5p'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
three_p = tf.RaggedTensor.from_value_rowids(D['seq_3p'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
ref = tf.RaggedTensor.from_value_rowids(D['seq_ref'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
alt = tf.RaggedTensor.from_value_rowids(D['seq_alt'][indexes].astype(np.int32), D['sample_idx'][indexes], nrows=len(samples['classes']))
strand = tf.RaggedTensor.from_value_rowids(D['strand_emb'][indexes].astype(np.float32), D['sample_idx'][indexes], nrows=len(samples['classes']))

cancer_strat = np.zeros_like(samples['classes']) ##no cancer info in this simulated data
y_label = np.stack(np.concatenate([samples['times'][:, np.newaxis], samples['censor'][:, np.newaxis], cancer_strat[:, np.newaxis]], axis=-1))
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(samples['classes'], y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(samples['classes'], y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))

tfds_all = tf.data.Dataset.from_tensor_slices(((five_p, three_p, ref, alt, strand), y_label))
tfds_all = tfds_all.batch(len(y_label), drop_remainder=False)

histories = []
evaluations = []
weights = []

cancer_test_ranks = {}
cancer_test_indexes = {}
cancer_test_expectation_ranks = {}

for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]
    train_data = (tf.gather(five_p, idx_train), tf.gather(three_p, idx_train), tf.gather(ref, idx_train), tf.gather(alt, idx_train), tf.gather(strand, idx_train))
    valid_data = (tf.gather(five_p, idx_valid), tf.gather(three_p, idx_valid), tf.gather(ref, idx_valid), tf.gather(alt, idx_valid), tf.gather(strand, idx_valid))
    test_data = (tf.gather(five_p, idx_test), tf.gather(three_p, idx_test), tf.gather(ref, idx_test), tf.gather(alt, idx_test), tf.gather(strand, idx_test))

    tfds_train = tf.data.Dataset.from_tensor_slices((train_data, y_label[idx_train]))
    tfds_train = tfds_train.shuffle(len(y_label), reshuffle_each_iteration=True).batch(250, drop_remainder=True)

    tfds_valid = tf.data.Dataset.from_tensor_slices((valid_data, y_label[idx_valid]))
    tfds_valid = tfds_valid.batch(len(idx_valid), drop_remainder=False)

    tfds_test = tf.data.Dataset.from_tensor_slices((test_data, y_label[idx_test]))
    tfds_test = tfds_test.batch(len(idx_test), drop_remainder=False)
    X = False
    while X == False:
        try:
            tile_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])
            mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model], output_dim=1, pooling='sum')
            losses = [RaggedModels.losses.CoxPH()]
            mil.model.compile(loss=losses,
                              metrics=[RaggedModels.losses.CoxPH()],
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                            ))
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_coxph', min_delta=0.0001, patience=20, mode='min', restore_best_weights=True)]
            history = mil.model.fit(tfds_train, validation_data=tfds_valid, epochs=10000, callbacks=callbacks)
            evaluation = mil.model.evaluate(tfds_test)
            histories.append(history.history)
            evaluations.append(evaluation)
            weights.append(mil.model.get_weights())
            y_pred_all = mil.model.predict(tfds_all)
            X = True
        except:
            del mil
            del tile_encoder
    ##get ranks per cancer
    for index, cancer in enumerate(['NA']):
        mask = np.where(cancer_strat == index)[0]
        cancer_test_indexes[cancer] = cancer_test_indexes.get(cancer, []) + [mask[np.isin(mask, idx_test, assume_unique=True)]]
        temp = np.exp(-y_pred_all[mask, 0]).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(mask))
        cancer_test_ranks[cancer] = cancer_test_ranks.get(cancer, []) + [ranks[np.isin(mask, idx_test, assume_unique=True)]]
    del mil
    del tile_encoder


indexes = np.concatenate(cancer_test_indexes['NA'])
ranks = np.concatenate(cancer_test_ranks['NA'])
concordance_index(samples['times'][indexes], ranks, samples['censor'][indexes])





# tfds_train = tfds_train.batch(len(idx_train), drop_remainder=True)

# predictions = mil.model.predict(tfds_train)
# from lifelines.utils import concordance_index
# concordance_index(samples['times'][idx_train], np.exp(-predictions), samples['censor'][idx_train])
# concordance_index(samples['times'], np.exp(-1 * samples['classes']), samples['censor'])


# with open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'instance_model_sum.pkl', 'wb') as f:
#     pickle.dump([evaluations, histories, weights], f)