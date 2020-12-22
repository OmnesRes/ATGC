import pylab as plt
import numpy as np
import pickle
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'sim_data' / 'sample_info' / 'experiment_2' / 'sim_data.pkl', 'rb'))

# instance_sum_evaluations, instance_sum_histories, weights = pickle.load(open(cwd / 'sim_data' / 'regression' / 'experiment_1' / 'instance_model_sum.pkl', 'rb'))
# instance_mean_evaluations, instance_mean_histories, weights = pickle.load(open(cwd / 'sim_data' / 'regression' / 'experiment_1' / 'instance_model_mean.pkl', 'rb'))
# sample_sum_evaluations, sample_sum_histories, weights = pickle.load(open(cwd / 'sim_data' / 'regression' / 'experiment_1' / 'sample_model_sum.pkl', 'rb'))
# sample_mean_evaluations, sample_mean_histories, weights = pickle.load(open(cwd / 'sim_data' / 'regression' / 'experiment_1' / 'sample_model_mean.pkl', 'rb'))


##sample characteristics
# sizes = np.unique(D['sample_idx'], return_counts=True)
# plt.hist(sizes[1], bins=100)
# plt.show()
##values

for i in range(1, 11):
    counts = []
    for index in np.where(np.array(samples['type']) == i)[0]:
        variants = D['class'][np.where(D['sample_idx'] == index)]
        counts.append(len(np.where(variants !=0)[0]))
    plt.scatter(counts, np.array(samples['values'])[np.array(samples['type']) == i])
plt.show()


##plot training
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1.0,
bottom=0.315,
left=0.07,
right=1.0,
hspace=0.2,
wspace=0.2)

for i in instance_mean_histories:
    ax.plot(i['val_categorical_crossentropy'][:-100], color='#1f77b4', label='I Mean')
for i in instance_sum_histories:
    ax.plot(i['val_categorical_crossentropy'][:-100], color='#1f77b4', linestyle='--', label='I Sum')
for i in sample_mean_histories:
    ax.plot(i['val_categorical_crossentropy'][:-50], color='#ff7f0e', label='S Mean')
for i in sample_sum_histories:
    ax.plot(i['val_categorical_crossentropy'][:-50], color='#ff7f0e', linestyle='--', label='S Sum')
plt.legend()
ax.set_yscale('log')
plt.show()