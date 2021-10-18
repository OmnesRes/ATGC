import pylab as plt
import numpy as np
import pickle
from sklearn.metrics import r2_score, confusion_matrix
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

tables, purities = pickle.load(open(cwd / 'figures' / 'vaf' / 'sim_data.pkl', 'rb'))
idx_test = pickle.load(open(cwd / 'figures' / 'vaf' / 'idx_test.pkl', 'rb'))
clonesig_predictions = pickle.load(open(cwd / 'figures' / 'vaf' / 'clonesig_predictions.pkl', 'rb'))
y_0 = np.array([len(tables[idx].clone.unique()) for idx in idx_test])

clonesig_matrix = confusion_matrix(y_0, clonesig_predictions)


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

myblue = make_colormap({0: '#ffffff', .1:'#AFDBF5', 1: '#4169E1'})

labels = ['1', '2', '3', '4']
matrix = clonesig_matrix[:-1, :-1]
precisions = np.diag(matrix / matrix.sum(axis=0))
recalls = np.diag(matrix / matrix.sum(axis=1))
f, axs = plt.subplots()
figure_matrix = axs.imshow(matrix / matrix.sum(axis=-1, keepdims=True), cmap=myblue, vmin=0, vmax=1)
cb = plt.colorbar(figure_matrix, ax=axs, pad=0.15)
cb.set_label(label='Fraction of True Class', rotation=270, labelpad=20)
f.subplots_adjust(top=0.91,
bottom=0.085,
left=0.11,
right=1.0,
hspace=0.2,
wspace=0.2)
axs = [axs, f.add_axes(axs.get_position())]
axs[1].set_facecolor('none')
axs[1].yaxis.tick_right()
axs[1].xaxis.tick_top()
axs[1].xaxis.set_label_position('top')
axs[1].yaxis.set_label_position('right')

axs[1].imshow(matrix / matrix.sum(axis=-1, keepdims=True), cmap=myblue, vmin=0, vmax=1)
ticks = list(range(4))
x_ticklabels = [labels, [str(round(i, 2)) for i in precisions]]
y_ticklabels = [labels, [str(round(i, 2)) for i in recalls]]
x_labels = ['Predicted Clones', 'Precision']
y_labels = ['True Clones', 'Recall']
for index, ax in enumerate(axs):
    [s.set_visible(False) for s in ax.spines.values()]
    ax.yaxis.set_inverted(True)
    ax.set(yticks=ticks, yticklabels=y_ticklabels[index])
    ax.set(xticks=ticks, xticklabels=x_ticklabels[index])
    ax.tick_params(axis="both", width=0, length=0, labelsize=10)
    ax.tick_params(axis="y", width=0, length=0, pad=10)

    ax.set_xlabel(x_labels[index], fontsize=14)
    if index == 1:
        ax.set_ylabel(y_labels[index], fontsize=14, rotation=270, labelpad=12)
    else:
        ax.set_ylabel(y_labels[index], fontsize=14)
plt.savefig(cwd / 'figures'/ 'vaf'/ 'clonesig_confusion.png', dpi=600)