import numpy as np
import pickle
import pylab as plt
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]


def make_colormap(colors):
    from matplotlib.colors import LinearSegmentedColormap, ColorConverter
    z = np.sort(list(colors.keys()))
    anchors = (z - min(z)) / (max(z) - min(z))
    CC = ColorConverter()
    R, G, B = [], [], []
    for i in range(len(z)):
        Ci = colors[z[i]]
        RGB = CC.to_rgb(Ci)
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])
    cmap_dict = {}
    cmap_dict['red'] = [(anchors[i], R[i], R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(anchors[i], G[i], G[i]) for i in range(len(G))]
    cmap_dict['blue'] = [(anchors[i], B[i], B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap', cmap_dict)
    return mymap


labels = ['other', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'TP53', 'PTEN', 'KRAS']
matrix = pickle.load(open(cwd / 'figures' / 'controls' / 'instances' / 'position' / 'results' / 'matrix.pkl', 'rb'))


fig = plt.figure()
gs = fig.add_gridspec(2, 4, height_ratios=[1, 10], width_ratios=[30, 3, 6, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 3])
ax4 = fig.add_subplot(gs[1, 1])
fig.subplots_adjust(top=0.928,
bottom=0.143,
left=0.138,
right=0.927,
hspace=0.088,
wspace=0.0)
myblue = make_colormap({0: '#ffffff', .1:'#AFDBF5', 1: '#4169E1'})
figure_matrix = ax2.imshow(matrix / matrix.sum(axis=-1, keepdims=True), cmap=myblue, vmin=0, vmax=1)
ax2.tick_params(width=0, labelsize=10, pad=0)
[s.set_visible(False) for s in ax2.spines.values()]
ax2.set_xticks(list(range(len(labels))))
ax2.set_xticklabels(labels, rotation=270)
ax2.set_yticks(list(range(len(labels))))
ax2.set_yticklabels(labels)
ax2.set_ylabel("True Class", fontsize=14, labelpad=10)
ax2.set_xlabel("Predicted Class", fontsize=14, labelpad=5)
ax1.bar(list(range(matrix.shape[0])), np.diag(matrix / matrix.sum(axis=0)), width=1, align='edge', edgecolor='k', linewidth=.5)
ax1.set_xlim(-.7, matrix.shape[0]+.7)
ax1.set_xticks([])
ax1.set_yticks([0, .5, 1])
ax1.set_yticklabels(['0', '50', '100'])
ax1.set_title('Precision', fontsize=12)
ax1.tick_params(length=3, width=.5, labelsize=6, pad=4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(.5)
ax1.spines['left'].set_position(['outward', -10])
ax1.spines['left'].set_bounds(0, 1)
ax1.spines['bottom'].set_visible(False)
ax4.barh(list(range(matrix.shape[0])), np.diag(matrix / matrix.sum(axis=1))[::-1], height=1, align='edge', edgecolor='k', linewidth=.5)
ax4.set_ylim(0, matrix.shape[0])
ax4.set_xticks([0, .5, 1])
ax4.set_yticks([])
ax4.set_xticklabels(['0', '50', '100'])
ax4.set_ylabel('Recall', fontsize=12, rotation=270, labelpad=20)
ax4.yaxis.set_label_position('right')
ax4.tick_params(length=3, width=.5, labelsize=6, pad=4)
ax4.spines['bottom'].set_linewidth(.5)
ax4.spines['bottom'].set_position(['outward', 8])
ax4.spines['bottom'].set_bounds(0, 1)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['top'].set_visible(False)
cbar = fig.colorbar(figure_matrix, cax=ax3, ticks=[0, .2, .4, .6, .8, 1])
plt.savefig(cwd / 'figures' / 'controls' / 'instances' / 'position' / 'position.png', dpi=600)

