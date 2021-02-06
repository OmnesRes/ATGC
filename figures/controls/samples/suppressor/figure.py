import pylab as plt
import numpy as np
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]

with open('figures/controls/samples/suppressor/results/latent.pkl', 'rb') as f:
    latent = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(np.concatenate(latent[0]).flat, bins=200, color='#1f77b4', edgecolor='#1f77b4', align='mid', linewidth=2)
ax.set_yscale('log', basey=10)
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