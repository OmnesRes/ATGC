import pylab as plt
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]


##panel plot
# panels = pickle.load(open(cwd / 'files' / 'tcga_panel_table.pkl', 'rb'))
# panel_cds = panels['cds'].values
# panel_exon = panels['exon'].values
# panel_intron = panels['total'].values - panels['exon'].values
#
# to_use = panels['Panel'].isin(['DUKE-F1-DX1', 'MSK-IMPACT341', 'MSK-IMPACT468', 'CRUK-TS', 'DFCI-ONCOPANEL-3', 'MDA-409-V1', 'UHN-555-V1', 'VICC-01-R2'])
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# fig.subplots_adjust(top=1.0,
# bottom=0.315,
# left=0.07,
# right=1.0,
# hspace=0.2,
# wspace=0.2)
#
# order = np.argsort(panel_exon[to_use])
# x = np.arange(3, (sum(to_use) + 1) * 3, 3)
# width = .7
# ax.bar(x - width, panel_exon[to_use][order], width, label='Exon', color='#1f77b4')
# ax.bar(x, panel_cds[to_use][order], width, label='CDS', color='#ff7f0e')
# ax.bar(x + width, panel_intron[to_use][order], width, label='Intron', color='#2ca02c')
# ax.set_yticks(np.arange(0, 2, .25) * 1e6)
# ax.set_yticklabels([str(i) for i in np.arange(0, 2, .25)], fontsize=10)
# ax.set_xticks(np.arange(3, (sum(to_use) + 1) * 3, 3))
# ax.set_xticklabels(panels['Panel'].values[to_use][order], rotation=90, fontsize=10)
# ax.set_ylim(0, panels.loc[panels['Panel'] == 'VICC-01-R2'].exon.values[0])
# ax.set_xlim(0, x[-1] + 1.4)
# ax.set_ylabel('Mb', fontsize=16)
# ax.tick_params(which='both', length=0)
# ax.tick_params(which='both', axis='y', pad=-10)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.legend(frameon=False, loc=(.05, .8))
#
# plt.savefig(cwd / 'figures' / 'tmb' / 'panel_sizes.pdf')



#predictions
results = pickle.load(open(cwd / 'figures' / 'tmb' / 'tcga' / 'MSK_468' / 'results' / 'predictions.pkl', 'rb'))

##residuals
def model(p, X, y=None, alpha=.1):
    _p = p.reshape((3, -1))
    y_hat = np.matmul(X, _p[0, :-1][:, np.newaxis])[:, 0] + _p[0, -1]
    y_bounds = np.matmul(X, _p[1:, :-1].T) + _p[1:, -1][np.newaxis, :]
    if y is None:
        return y_hat, y_bounds
    else:
        # residuals
        residuals = y - y_hat
        # get MAD fit
        quantiles = np.array([0.5])
        loss = np.mean(residuals * (quantiles[np.newaxis, :] - (residuals < 0)))
        # add loss for bounds
        residuals = y[:, np.newaxis] - y_bounds
        quantiles = np.array([(alpha / 2), 1 - (alpha / 2)])
        # return check function with quantiles - aka quantile loss
        loss += np.mean(residuals * (quantiles[np.newaxis, :] - (residuals < 0)))
        return loss

d = 4
pf = PolynomialFeatures(degree=d, include_bias=False)
x0 = np.random.normal(0, 1, (3, d + 1)).flatten()


residuals = []
x_preds = []
y_pred_bounds = []

results['y_true'] = results['y_true'][:, 1]
for i in ['counting', 'naive', 'position', 'sequence']:
    if i != 'counting':
        results[i] = results[i][:, 1]
    res = minimize(model, x0, args=(pf.fit_transform(results[i][:, np.newaxis]), results['y_true']))
    x_pred = np.linspace(np.min(results[i]), np.max(results[i]), 200)
    y_pred, temp_y_pred_bounds = model(res.x, pf.fit_transform(x_pred[:, np.newaxis]))
    x_preds.append(x_pred)
    y_pred_bounds.append(temp_y_pred_bounds)


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['Counting', 'ATGC', 'ATGC + Pos', 'ATGC + Seq']
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
fig.subplots_adjust(top=1.0,
bottom=0.085,
left=0.08,
right=0.99,
hspace=0.11,
wspace=0.2)


for index, (axis, data, label) in enumerate(zip([ax1, ax2, ax3, ax4], ['counting', 'naive', 'position', 'sequence'], labels)):
    axis.scatter(results[data], results['y_true'], s=5, edgecolor='none', alpha=.15, color=colors[index])
    axis.fill_between(x_preds[index], y_pred_bounds[index][:, 0], y_pred_bounds[index][:, 1], alpha=.2, color=colors[index])
    axis.plot(list(range(0, 65)), list(range(0, 65)), color='k', lw=1, alpha=.5)
    axis.set_xticks([np.log(i+1) for i in [0, 1, 2, 3, 5, 10, 25, 64]])
    axis.set_xticklabels(['0', '1', '2', '3', '5', '10', '25', '64'], fontsize=9)
    axis.set_yticks([np.log(i + 1) for i in [0, 1, 2, 3, 5, 10, 25, 64]])
    axis.set_yticklabels(['0', '1', '2', '3', '5', '10', '25', '64'], fontsize=9)
    if index % 2 == 0:
        axis.set_ylabel('WES TMB')
    if index > 1:
        axis.set_xlabel('Predicted TMB')
    axis.set_xlim(np.log(0 + 1) - .3, np.log(1 + 64) + .1)
    axis.set_ylim(np.log(0 + 1) - .3, np.log(1 + 64) + .5)
    axis.set_title(label, y=.85)
    axis.tick_params(length=3, width=1)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['left'].set_bounds(np.log(1 + 0), np.log(1 + 64))
    axis.spines['bottom'].set_bounds(np.log(1 + 0), np.log(1 + 64))

plt.savefig(cwd / 'figures' / 'tmb' / 'pred_true.png', dpi=600)


