from figures.controls.samples.sim_data.sim_data_tools import *
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]
def generate_sample(mean_variants=[5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300],
                    positive_choices=None):
    center = np.random.choice(mean_variants, 1)
    total_count = int(np.random.normal(center, int(np.ceil(center * .2))))
    if total_count < 1:
        total_count *= -1
    if total_count == 0:
        total_count = np.random.choice([2, 3, 4, 5, 6], 1)

    positive_count = int(np.ceil(np.random.random() * total_count))
    control_count = total_count - positive_count * len(positive_choices)

    control_count = max(control_count, 0)
    positive_variants = []
    positive_instances = []

    control_variants = [generate_variant() for i in range(control_count)]
    while True:
        y = False
        for i in control_variants:
            if check_variant(i, positive_choices):
                print('checked')
                y = True
                break
        if y:
            control_variants = [generate_variant() for i in range(control_count)]
        else:
            break

    for index, i in enumerate(positive_choices):
        for ii in range(positive_count):
            positive_variants.append(i)
            positive_instances.append(index + 1)

    sample_value = np.random.normal(positive_count, positive_count / 10)
    if sample_value < 0:
        sample_value *= -1
    return [control_variants + positive_variants, [0] * len(control_variants) + positive_instances, sample_value]

##dictionary for instance level data
instances = {'sample_idx': [],
                 'seq_5p': [],
                 'seq_3p': [],
                  'seq_ref': [],
                  'seq_alt': [],
                  'chr': [],
                  'pos_float': [],
                  'strand': [],
                  'cds': [],
                  'class': []}

##how many different variants you want to label a positive sample
positive_choices = [generate_variant() for i in range(1)]

samples = {'classes': [], 'values':[]}

for idx in range(1000):
    variants = generate_sample(positive_choices=positive_choices)
    samples['classes'] = samples['classes'] + [1]
    samples['values'] = samples['values'] + [variants[2]]
    instances['sample_idx'] = instances['sample_idx'] + [idx] * len(variants[0])
    instances['seq_5p'] = instances['seq_5p'] + [i[0] for i in variants[0]]
    instances['seq_3p'] = instances['seq_3p'] + [i[1] for i in variants[0]]
    instances['seq_ref'] = instances['seq_ref'] + [i[2] for i in variants[0]]
    instances['seq_alt'] = instances['seq_alt'] + [i[3] for i in variants[0]]
    instances['chr'] = instances['chr'] + [i[4] for i in variants[0]]
    instances['pos_float'] = instances['pos_float'] + [i[5] for i in variants[0]]
    instances['strand'] = instances['strand'] + [i[6] for i in variants[0]]
    instances['cds'] = instances['cds'] + [0 for i in variants[0]]
    instances['class'] = instances['class'] + variants[1]

for i in instances:
    instances[i] = np.array(instances[i])

nucleotide_mapping = {'-': 0, 'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
instances['seq_5p'] = np.stack(np.apply_along_axis(lambda x: np.array([nucleotide_mapping[i] for i in x]), -1, instances['seq_5p']), axis=0)
instances['seq_3p'] = np.stack(np.apply_along_axis(lambda x: np.array([nucleotide_mapping[i] for i in x]), -1, instances['seq_3p']), axis=0)
instances['seq_ref'] = np.stack(np.apply_along_axis(lambda x: np.array([nucleotide_mapping[i] for i in x]), -1, instances['seq_ref']), axis=0)
instances['seq_alt'] = np.stack(np.apply_along_axis(lambda x: np.array([nucleotide_mapping[i] for i in x]), -1, instances['seq_alt']), axis=0)


variant_encoding = np.array([0, 2, 1, 4, 3])
instances['seq_5p'] = np.stack([instances['seq_5p'], variant_encoding[instances['seq_3p'][:, ::-1]]], axis=2)
instances['seq_3p'] = np.stack([instances['seq_3p'], variant_encoding[instances['seq_5p'][:, :, 0][:, ::-1]]], axis=2)
t = instances['seq_ref'].copy()
i = t != 0
t[i] = variant_encoding[instances['seq_ref'][:, ::-1]][i[:, ::-1]]
instances['seq_ref'] = np.stack([instances['seq_ref'], t], axis=2)
t = instances['seq_alt'].copy()
i = t != 0
t[i] = variant_encoding[instances['seq_alt'][:, ::-1]][i[:, ::-1]]
instances['seq_alt'] = np.stack([instances['seq_alt'], t], axis=2)
del i, t

with open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'regression' / 'experiment_1' / 'sim_data.pkl', 'wb') as f:
    pickle.dump([instances, samples, ], f)




#
# import pylab as plt
# sample_counts = []
# for i in range(len(samples['values'])):
#     indexes = np.where(instances['sample_idx'] == i)
#     sample_counts.append(sum(instances['class'][indexes]))
#
# plt.scatter(sample_counts, samples['values'], s=.1)
# # plt.hist(samples['values'], bins=100)
#
# plt.show()