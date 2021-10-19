import pylab as plt
import numpy as np
from matplotlib import cm
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))


with open(cwd / 'figures' / 'controls' / 'samples' / 'hotspot' / 'results' / 'latent_braf.pkl', 'rb') as f:
    latent, labels = pickle.load(f)

paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]

fold = 0
fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, patches = ax.hist(np.array([j for i in latent[fold] for j in i])[labels[fold]], bins=200, range=(0, 1), edgecolor=paired[2], color=paired[2], align='mid', linewidth=1)
ax.hist(np.array([j for i in latent[fold] for j in i])[labels[fold]], bins=bins, edgecolor=paired[3], color=paired[3], align='mid', linewidth=1)
ax.set_yscale('log', base=10)
ax.set_ylim(1, 10**6)
ax.set_xlim(0, 1)
ax.spines['bottom'].set_bounds(0, 1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(which='both', width=0, axis='y')
ax.tick_params(which='both', width=1, axis='x')
ax.set_ylabel("Count", fontsize=14)
ax.set_xlabel("Attention", fontsize=14)
ax.set_title('BRAF V600', fontsize=22)
plt.savefig(cwd / 'figures' / 'controls' / 'samples' / 'hotspot' / 'latent_braf.pdf')