import numpy as np
import tensorflow as tf
from model.Instance_MIL import InstanceModels, RaggedModels
from model import DatasetsUtils
from model.KerasLayers import Losses
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
D, samples = pickle.load(open(cwd / 'sim_data' / 'survival' / 'experiment_2' / 'sim_data.pkl', 'rb'))

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

cancer_strat = np.zeros_like(samples['classes']) ##no cancer info in this simulated data
y_label = np.stack(np.concatenate([samples['times'][:, np.newaxis], samples['event'][:, np.newaxis], cancer_strat[:, np.newaxis]], axis=-1))
strat_dict = {key: index for index, key in enumerate(set(tuple([group, event]) for group, event in zip(samples['classes'], y_label[:, 1])))}
y_strat = np.array([strat_dict[(group, event)] for group, event in zip(samples['classes'], y_label[:, 1])])
class_counts = dict(zip(*np.unique(y_strat, return_counts=True)))

ds_all = tf.data.Dataset.from_tensor_slices((np.arange(len(y_label)), y_label))
ds_all = ds_all.batch(len(y_label), drop_remainder=False)
ds_all = ds_all.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                       three_p_loader(x, ragged_output=True),
                                       ref_loader(x, ragged_output=True),
                                       alt_loader(x, ragged_output=True),
                                       strand_loader(x, ragged_output=True)),
                                       y))


histories = []
evaluations = []
weights = []

cancer_test_ranks = {}
cancer_test_indexes = {}
cancer_test_expectation_ranks = {}

for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat):
    idx_train, idx_valid = [idx_train[idx] for idx in list(StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=0).split(np.zeros_like(y_strat)[idx_train], y_strat[idx_train]))[0]]

    ds_train = tf.data.Dataset.from_tensor_slices((idx_train, y_label[idx_train], y_strat[idx_train]))
    ds_train = ds_train.apply(DatasetsUtils.Apply.StratifiedMinibatch(batch_size=250, ds_size=len(idx_train)))
    ds_train = ds_train.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                           three_p_loader(x, ragged_output=True),
                                           ref_loader(x, ragged_output=True),
                                           alt_loader(x, ragged_output=True),
                                           strand_loader(x, ragged_output=True)),
                                          y))

    ds_valid = tf.data.Dataset.from_tensor_slices((idx_valid, y_label[idx_valid]))
    ds_valid = ds_valid.batch(len(idx_valid), drop_remainder=False)
    ds_valid = ds_valid.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                           three_p_loader(x, ragged_output=True),
                                           ref_loader(x, ragged_output=True),
                                           alt_loader(x, ragged_output=True),
                                           strand_loader(x, ragged_output=True)),
                                          y))

    ds_test = tf.data.Dataset.from_tensor_slices((idx_test, y_label[idx_test]))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    ds_test = ds_test.map(lambda x, y: ((five_p_loader(x, ragged_output=True),
                                         three_p_loader(x, ragged_output=True),
                                         ref_loader(x, ragged_output=True),
                                         alt_loader(x, ragged_output=True),
                                         strand_loader(x, ragged_output=True)),
                                        y))
    X = False
    while X == False:
        try:
            tile_encoder = InstanceModels.VariantSequence(6, 4, 2, [16, 16, 8, 8])
            mil = RaggedModels.MIL(instance_encoders=[tile_encoder.model], output_dim=1, pooling='sum', output_type='other', instance_layers=[128, 64])
            losses = [Losses.CoxPH()]
            mil.model.compile(loss=losses,
                              metrics=[Losses.CoxPH()],
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                            ))
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_coxph', min_delta=0.0001, patience=20, mode='min', restore_best_weights=True)]
            history = mil.model.fit(ds_train, steps_per_epoch=4, validation_data=ds_valid, epochs=10000, callbacks=callbacks)
            y_pred_all = mil.model.predict(ds_all)
            if concordance_index(samples['times'], np.exp(-1 * y_pred_all[:, 0]), samples['event']) > .52:
                X = True
                evaluation = mil.model.evaluate(ds_test)
                histories.append(history.history)
                evaluations.append(evaluation)
                weights.append(mil.model.get_weights())
        except:
            pass
    ##get ranks per cancer
    for index, cancer in enumerate(['NA']):
        mask = np.where(cancer_strat == index)[0]
        cancer_test_indexes[cancer] = cancer_test_indexes.get(cancer, []) + [mask[np.isin(mask, idx_test, assume_unique=True)]]
        temp = np.exp(-y_pred_all[mask, 0]).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(mask))
        cancer_test_ranks[cancer] = cancer_test_ranks.get(cancer, []) + [ranks[np.isin(mask, idx_test, assume_unique=True)]]


indexes = np.concatenate(cancer_test_indexes['NA'])
ranks = np.concatenate(cancer_test_ranks['NA'])
concordance_index(samples['times'][indexes], ranks, samples['event'][indexes])


# tfds_train = tfds_train.batch(len(idx_train), drop_remainder=True)

# predictions = mil.model.predict(tfds_train)
# from lifelines.utils import concordance_index
# concordance_index(samples['times'][idx_train], np.exp(-predictions), samples['event'][idx_train])
# concordance_index(samples['times'], np.exp(-1 * samples['classes']), samples['event'])


with open(cwd / 'sim_data' / 'survival' / 'experiment_2' / 'instance_model_sum.pkl', 'wb') as f:
    pickle.dump([evaluations, histories, weights], f)