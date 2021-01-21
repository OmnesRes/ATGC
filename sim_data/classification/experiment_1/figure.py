import pylab as plt
import numpy as np
import pickle
import pathlib
from matplotlib import cm
path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sim_data.pkl', 'rb'))

instance_sum_evaluations, instance_sum_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'instance_model_sum.pkl', 'rb'))
instance_mean_evaluations, instance_mean_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'instance_model_mean.pkl', 'rb'))
sample_sum_evaluations, sample_sum_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_sum.pkl', 'rb'))
sample_mean_evaluations, sample_mean_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_mean.pkl', 'rb'))
sample_sum_attention_evaluations, sample_sum_attention_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_attention_sum.pkl', 'rb'))
sample_mean_attention_evaluations, sample_mean_attention_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_attention_mean.pkl', 'rb'))
sample_both_attention_evaluations, sample_both_attention_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_attention_both.pkl', 'rb'))
sample_dynamic_attention_evaluations, sample_dynamic_attention_histories, weights = pickle.load(open(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'sample_model_attention_dynamic.pkl', 'rb'))



losses = np.array([i[-1] for i in instance_mean_evaluations + instance_sum_evaluations + \
                   sample_mean_evaluations + sample_sum_evaluations + \
                   sample_mean_attention_evaluations + sample_sum_attention_evaluations + \
                   sample_both_attention_evaluations + sample_dynamic_attention_evaluations])
losses = losses / max(losses)


metrics = [-i[1] for i in instance_mean_evaluations + instance_sum_evaluations + \
                   sample_mean_evaluations + sample_sum_evaluations + \
                   sample_mean_attention_evaluations + sample_sum_attention_evaluations + \
                   sample_both_attention_evaluations + sample_dynamic_attention_evaluations]


epochs = np.array([len(i['val_categorical_crossentropy']) - 40 for i in instance_mean_histories + instance_sum_histories]
                  + [len(i['val_categorical_crossentropy']) - 20 for i in sample_mean_histories + sample_sum_histories +\
                     sample_mean_attention_histories + sample_sum_attention_histories +\
                     sample_both_attention_histories + sample_dynamic_attention_histories])
epochs = epochs / max(epochs)


paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
colors = [paired[0]] * 3 + [paired[1]] * 3 + [paired[2]] * 3 + [paired[3]] * 3 + [paired[4]] * 3 + [paired[5]] * 3 + [paired[6]] * 3 + [paired[7]] * 3

spacer = np.ones_like(losses)/25
centers = np.concatenate([np.arange(3) + i * 3.2 for i in range(8)])
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=1.0,
bottom=0.0,
left=0.05,
right=0.945,
hspace=0.2,
wspace=0.2)
ax.bar(centers, losses, edgecolor='k', bottom=spacer, color=colors, align='center', linewidth=.5, width=1)

ax.set_xlim(min(centers) - .51, max(centers) + .51)
ax.set_ylim(-1.003, max(losses + spacer) + .003)
ax.set_yticks([])
ax.set_xticks([])


ax2 = ax.twinx()
ax2.bar(centers, metrics, edgecolor='k', color=colors, align='center', linewidth=.5, width=1)
ax2.set_ylim(-1.003, max(losses + spacer) + .003)
ax2.set_xlim(min(centers) - .51, max(centers) + .51)
ax2.set_yticks([])
ax2.set_xticks([])

ax.set_ylabel(' ' * 19 + 'Losses', fontsize=24, labelpad=0)
ax2.set_ylabel(' ' * 19 + 'Accuracy', rotation=-90, fontsize=24, labelpad=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
plt.savefig(cwd / 'sim_data' / 'classification' / 'experiment_1' / 'figure.pdf')


