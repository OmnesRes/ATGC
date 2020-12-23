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

D, samples = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_2' / 'sim_data.pkl', 'rb'))

instance_sum_evaluations, instance_sum_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_2' / 'instance_model_sum.pkl', 'rb'))
instance_mean_evaluations, instance_mean_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_2' / 'instance_model_mean.pkl', 'rb'))
sample_sum_evaluations, sample_sum_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_2' / 'sample_model_sum.pkl', 'rb'))
sample_mean_evaluations, sample_mean_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_2' / 'sample_model_mean.pkl', 'rb'))

losses = np.array([i[-1] for i in instance_sum_evaluations + instance_mean_evaluations + sample_sum_evaluations + sample_mean_evaluations])
losses = losses / max(losses)
first_loss = losses[np.arange(0, len(losses), 3)]
second_loss = losses[np.arange(1, len(losses), 3)]
third_loss = losses[np.arange(2, len(losses), 3)]

epochs = np.array([len(i['val_categorical_crossentropy']) - 40 for i in instance_sum_histories + instance_mean_histories]
                  + [len(i['val_categorical_crossentropy']) - 20 for i in sample_sum_histories + sample_mean_histories])
epochs = epochs / max(epochs)
first_epoch = epochs[np.arange(0, len(epochs), 3)]
second_epoch = epochs[np.arange(1, len(epochs), 3)]
third_epoch = epochs[np.arange(2, len(epochs), 3)]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

spacer = np.ones_like(first_loss)/20
# centers = np.concatenate([np.arange(3) + i * 4 for i in range(4)])
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=1.0,
bottom=0.0,
left=0.0605,
right=0.995,
hspace=0.2,
wspace=0.2)
ax.bar(list(range(4)), first_loss, edgecolor='k', bottom=spacer, color=colors, align='center', linewidth=.5, width=1)
ax.bar(list(range(4)), second_loss, bottom=first_loss + spacer, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)
ax.bar(list(range(4)), third_loss, bottom=first_loss + second_loss + spacer, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)

ax.bar(list(range(4)), -first_epoch, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)
ax.bar(list(range(4)), -second_epoch, bottom=-first_epoch, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)
ax.bar(list(range(4)), -third_epoch, bottom=-first_epoch - second_epoch, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)

ax.set_xlim(-.503, 3.503)
ax.set_ylim(-max(first_epoch + second_epoch + third_epoch) - .003, max(first_loss + second_loss + third_loss + spacer) + .003)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel(' Epochs' + ' ' * 12 + 'Losses', fontsize=24)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.savefig(cwd / 'sim_data' / 'classification' / 'experiment_2' / 'figure.pdf')


