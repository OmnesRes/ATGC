from figures.controls.samples.sim_data.sim_data_tools import *
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]


##random witness rate
def generate_sample(mean_variants=[5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300],
                    control=True, positive_choices=None, fixed=['five_p']):

    center = np.random.choice(mean_variants, 1)
    total_count = int(np.random.normal(center, int(np.ceil(center * .2))))
    if total_count < 1:
        total_count *= -1
    if total_count == 0:
        total_count = np.random.choice([2, 3, 4, 5, 6], 1)

    if control:
        positive_counts = [10]
    else:
        positive_counts = [20]

    control_count = total_count - sum(positive_counts)

    control_count = max(control_count, 0)
    positive_variants = []
    positive_instances = []

    control_variants = [generate_variant() for i in range(control_count)]
    while True:
        ##this could be more efficient, replace offending variant with a checked variant
        y = False
        for i in control_variants:
            if check_variant(i, positive_choices, to_check=fixed):
                print('checked')
                y = True
                break
        if y:
            control_variants = [generate_variant() for i in range(control_count)]
        else:
            break

    for index, i in enumerate(positive_choices):
        for ii in range(positive_counts[index]):
            positive_variant = list(generate_variant())
            if 'five_p' in fixed:
                positive_variant[0] = i[0]
            if 'three_p' in fixed:
                positive_variant[1] = i[1]
            if 'ref' in fixed:
                positive_variant[2] = i[2]
            if 'alt' in fixed:
                positive_variant[3] = i[3]
            positive_variants.append(positive_variant)
            positive_instances.append(index + 1)

    return [control_variants + positive_variants, [0] * len(control_variants) + positive_instances]

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

samples = {'classes': []}

for idx in range(1000):
    ##what percent of samples are control
    if np.random.sample() < .5:
        variants = generate_sample(positive_choices=positive_choices)
        samples['classes'] = samples['classes'] + [0]
    else:
        variants = generate_sample(control=False, positive_choices=positive_choices)
        samples['classes'] = samples['classes'] + [1]
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

def get_context(five_p, three_p, ref, alt):
    if ref[0] == 'T' or ref[0] == 'C':
        return five_p[-1] + ref[0] + alt[0] + three_p[0]
    else:
        return str(Seq(three_p[0]).reverse_complement()) + str(Seq(ref[0]).reverse_complement()) + str(Seq(alt[0]).reverse_complement()) + str(Seq(five_p[-1]).reverse_complement())

instances['context'] = [get_context(i, j, k, l) for i,j,k,l in zip(instances['seq_5p'], instances['seq_3p'], instances['seq_ref'], instances['seq_alt'])]

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

with open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_3' / 'sim_data.pkl', 'wb') as f:
    pickle.dump([instances, samples, ], f)




