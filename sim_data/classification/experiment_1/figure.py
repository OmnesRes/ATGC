import pylab as plt
import numpy as np
import pickle
import math
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sim_data.pkl', 'rb'))

with open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_mean.pkl', 'rb') as f:
    history, weights = pickle.load(f)


##sample characteristics
sizes = np.unique(D['sample_idx'], return_counts=True)
plt.hist(sizes[1], bins=100)
plt.show()
##witness rate
rates = []
for index, i in enumerate(samples['classes']):
    if i == 1:
        variants = D['class'][np.where(D['sample_idx'] == index)]
        witness_rate = len(np.where(variants !=0)[0]) / len(variants)
        rates.append(witness_rate)

plt.hist(rates, bins=20)
plt.xlim(0, 1)
plt.show()


# mean_variants = [5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300]
# counts = []
# for i in range(1000):
#     center = np.random.choice(mean_variants, 1)
#     total_count = int(np.random.normal(center, int(math.ceil(center * .2))))
#     if total_count < 1:
#         total_count *= -1
#     if total_count == 0:
#         total_count = np.random.choice([2, 3, 4, 5, 6], 1)
#     counts.append(total_count)
# plt.hist(counts, bins=100)
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# fig.subplots_adjust(top=1.0,
# bottom=0.315,
# left=0.07,
# right=1.0,
# hspace=0.2,
# wspace=0.2)
#
# ax.plot(history['val_categorical_crossentropy'])
# ax.scatter(list(range(len(history['val_categorical_crossentropy']))), history['val_categorical_crossentropy'])
# ax.set_yscale('log')
# plt.show()