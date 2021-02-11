import pylab as plt
import numpy as np
from matplotlib import cm
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]

with open('figures/controls/samples/suppressor/results/latent.pkl', 'rb') as f:
    latent, labels = pickle.load(f)

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]

fold = 0

fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, patches = ax.hist(np.concatenate(latent[fold]).flat[~np.array(labels[fold])], bins=200, range=(0, 1), edgecolor=paired[0], color=paired[0], align='mid', linewidth=1)
ax.hist(np.concatenate(latent[fold]).flat[labels[fold]], bins=bins, edgecolor=paired[1], color=paired[1], align='mid', linewidth=1)
ax.set_yscale('log', base=10)
ax.set_ylim(1, 10**6)
ax.set_xlim(-.01, 1)
ax.spines['bottom'].set_bounds(0, 1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(which='both', width=0, length=0, axis='y')
ax.tick_params(which='both', width=1, axis='x')
ax.set_ylabel("Count", fontsize=14)
ax.set_xlabel("Attention", fontsize=14)
ax.set_title('PTEN Gene', fontsize=22)
plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'suppressor' / 'latent.pdf')