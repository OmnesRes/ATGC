import pylab as plt
import numpy as np
import pickle
import pathlib
from matplotlib import cm
from lifelines.utils import concordance_index

path = pathlib.Path.cwd()

if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
    import sys
    sys.path.append(str(cwd))

D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sim_data.pkl', 'rb'))

instance_sum_evaluations, instance_sum_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'instance_model_sum.pkl', 'rb'))
sample_sum_evaluations, sample_sum_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_sum.pkl', 'rb'))
sample_sum_attention_evaluations, sample_sum_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_sum.pkl', 'rb'))
sample_both_attention_evaluations, sample_both_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_both.pkl', 'rb'))
sample_dynamic_attention_evaluations, sample_dynamic_attention_histories, weights = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_dynamic.pkl', 'rb'))

instance_indexes, instance_ranks, instance_sample_dfs = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'instance_model_sum_eval.pkl', 'rb'))
sample_sum_indexes, sample_sum_ranks, sample_sum_sample_dfs = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_sum_eval.pkl', 'rb'))
sample_sum_attention_indexes, sample_sum_attention_ranks, sample_sum_attention_sample_dfs, attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_sum_eval.pkl', 'rb'))
sample_both_attention_indexes, sample_both_attention_ranks, sample_both_attention_sample_dfs, attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_both_eval.pkl', 'rb'))
sample_dynamic_attention_indexes, sample_dynamic_attention_ranks, sample_dynamic_attention_sample_dfs, attentions = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'sample_model_attention_dynamic_eval.pkl', 'rb'))

concordances = np.array([concordance_index(samples['times'][indexes], ranks, samples['event'][indexes]) for indexes, ranks in zip([instance_indexes, sample_sum_indexes, sample_sum_attention_indexes, sample_both_attention_indexes, sample_dynamic_attention_indexes], [instance_ranks, sample_sum_ranks, sample_sum_attention_ranks, sample_both_attention_ranks, sample_dynamic_attention_ranks])])
concordances = concordances / max(concordances)

epochs = np.array([len(i['val_coxph']) - 10 for i in instance_sum_histories + sample_sum_histories + sample_sum_attention_histories + sample_both_attention_histories + sample_dynamic_attention_histories])
epochs = epochs / max(epochs)

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]
colors_concordances = [paired[1], paired[3], paired[5], paired[7], paired[9]]

colors_epochs = [paired[1]] * 5 + [paired[3]] * 5 + [paired[5]] * 5 + [paired[7]] * 5 + [paired[9]] * 5

spacer = np.ones_like(concordances)/25
concordance_centers = [.5 + i * 1.1 for i in range(5)]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(
top=1.0,
bottom=0.0,
left=0.05,
right=0.945,
hspace=0.2,
wspace=0.2)
ax.bar(concordance_centers, concordances, edgecolor='k', bottom=spacer, color=colors_concordances, align='center', linewidth=.5, width=1)

ax.set_xlim(min(concordance_centers) - .503, max(concordance_centers) + .503)
ax.set_ylim(-max(epochs) - .003, max(concordances + spacer) + .003)
ax.set_yticks([])
ax.set_xticks([])

epoch_centers = np.concatenate([np.arange(.1, 1.1, .2) + i * 1.1 for i in range(5)])
ax2 = ax.twinx()
ax2.bar(epoch_centers, -epochs, edgecolor='k', color=colors_epochs, align='center', linewidth=.5, width=1/5)
ax2.set_ylim(-max(epochs) - .003, max(concordances + spacer) + .003)
ax2.set_xlim(min(epoch_centers) - .503, max(epoch_centers) + .503)
ax2.set_yticks([])
ax2.set_xticks([])

ax.set_ylabel(' ' * 19 + 'Concordances', fontsize=24, labelpad=0)
ax2.set_ylabel(' ' * 19 + 'Epochs', rotation=-90, fontsize=24, labelpad=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'survival' / 'experiment_1' / 'figure.pdf')


