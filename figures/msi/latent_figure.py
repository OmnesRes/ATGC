import pickle
import pylab as plt
import seaborn as sns
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]


with open(cwd / 'figures' / 'msi' / 'results' / 'latents.pkl', 'rb') as f:
    latents = pickle.load(f)


##choose one K fold
non_repeats = latents[6][0]
repeats = latents[6][1]


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
fig.subplots_adjust(top=0.975,
bottom=0.07,
left=0.11,
right=0.98,
hspace=0.14,
wspace=0.04)
sns.kdeplot(non_repeats.flatten(), shade=True, gridsize=300, ax=ax1, alpha=1)
sns.kdeplot(non_repeats.flatten(), shade=False, gridsize=300, ax=ax1, alpha=1, color='k', linewidth=1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1)
ax1.set_xticks([])
ax1.tick_params(axis='y', length=0, width=0, labelsize=8)
ax1.set_ylabel('Variant Density (thousand)', fontsize=10)
ax1.set_xlim(.04, .13)
ax1.set_title('Other', fontsize=12, loc='left', y=.87, x=.01)
sns.kdeplot(repeats.flatten(), shade=True, gridsize=300, ax=ax2, alpha=1)
sns.kdeplot(repeats.flatten(), shade=False, gridsize=300, ax=ax2, alpha=1, color='k', linewidth=1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_linewidth(1)
ax2.set_xticks([])
ax2.tick_params(axis='y', length=0, width=0, labelsize=8)
ax2.set_xlabel('Attention', fontsize=12)
ax2.set_ylabel('Variant Density (thousand)', fontsize=10, labelpad=8)
ax2.set_xlim(.04, .13)
ax2.set_title('Simple Repeats', fontsize=12, loc='left', y=.87, x=.01)
fig.canvas.draw()
ax1.set_yticklabels([str(int(round(float(i.get_text())/100 * non_repeats.shape[0] / 1000, 0))) for i in ax1.get_yticklabels()])
ax2.set_yticklabels([str(int(round(float(i.get_text())/100 * repeats.shape[0] / 1000, 0))) for i in ax2.get_yticklabels()])
plt.savefig(cwd / 'figures' / 'msi' / 'kde.pdf')





