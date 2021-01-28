import numpy as np
import pickle
from model.CustomKerasModels import InputFeatures, ATGC
from model.CustomKerasTools import BatchGenerator, Losses
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import r2_score, classification_report
import pylab as plt
disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[6], True)
tf.config.experimental.set_visible_devices(physical_devices[6], 'GPU')

D, maf = pickle.load(open('/home/janaya2/Desktop/ATGC_paper/figures/controls/data/data.pkl', 'rb'))
sample_df = pickle.load(open('files/tcga_sample_table.pkl', 'rb'))

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


with open('figures/controls/samples/suppressor/results/weights.pkl', 'rb') as f:
    weights = pickle.load(f)


test_idx = []
predictions = []
evaluations = []
for index, (idx_train, idx_test) in enumerate(StratifiedKFold(n_splits=8, random_state=0, shuffle=True).split(y_strat, y_strat)):

    atgc.mil_model.set_weights(weights[index])
    data_test = next(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                     y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=idx_test).data_generator())
    evaluations.append(atgc.mil_model.evaluate(data_test[0], data_test[1]))
    predictions.append(atgc.mil_model.predict(data_test[0])[0, :, :-1])
    test_idx.append(idx_test)

with open('figures/controls/samples/suppressor/results/predictions.pkl', 'wb') as f:
    pickle.dump([y_strat, test_idx, predictions], f)



atgc.mil_model.set_weights(weights[0])
latent = atgc.intermediate_model.predict(BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                                y_label=y_label, y_stratification=y_strat, y_weights=y_weights, sampling_approach=None, idx_sample=test_idx[0]).data_generator(), steps=1)


with open('figures/controls/samples/suppressor/results/latent.pkl', 'wb') as f:
    pickle.dump(latent, f)


# # # #
# print(classification_report(y_strat[np.concatenate(test_idx, axis=-1)], np.argmax(np.concatenate(predictions, axis=0), axis=-1), digits=4))
# # #
# #
# plt.hist(latent, bins=100)
# plt.ylim(0, 1000)
# plt.show()
#
# # #
# test=maf.iloc[np.concatenate(np.array([np.where(D['sample_idx'] == i)[0] for i in range(y_label.shape[0])])[test_idx[0]], axis=-1)]['Hugo_Symbol'][latent[:, 0] > .5]
#
# sorted(list(zip(latent[latent[:, 0] > .5],
#                 test,
#                 D['pos_bin'][np.concatenate(np.array([np.where(D['sample_idx'] == i)[0] for i in range(y_label.shape[0])])[test_idx[0]], axis=-1)][latent[:, 0] > .5],
#                 D['pos_loc'][:,0][np.concatenate(np.array([np.where(D['sample_idx'] == i)[0] for i in range(y_label.shape[0])])[test_idx[0]], axis=-1)][latent[:, 0] > .5])))
#



