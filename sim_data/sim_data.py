import numpy as np
import pickle
from Bio.Seq import Seq
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]

#helper function for generating a string of nucleotides of any length
def gen_ran_nuc(len_nuc):
    return ''.join(np.random.choice(['A', 'T', 'C', 'G'], len_nuc))

#generate a variant (5p, 3p, ref, alt, chromosome, position, strand) with a certain chance of being an indel
def generate_variant(length=6, indel_percent=.1):
    five_p = gen_ran_nuc(length)
    three_p = gen_ran_nuc(length)
    choices = ['A', 'T', 'C', 'G']
    ref = ''.join(np.random.choice(choices, 1))
    remaining_choices = choices.copy()
    remaining_choices.pop(choices.index(ref))
    alt = ''.join(np.random.choice(remaining_choices, 1))
    ref += '-----'
    alt += '-----'
    if np.random.sample() < indel_percent:
        size = int(np.random.choice(range(1, length + 1), 1))
        ##even chance of being insertion vs. deletion
        if np.random.sample() < .5:
            ref = '------'
            alt = ''.join(np.random.choice(choices, size)) + '-' * (length - size)
        else:
            ref = ''.join(np.random.choice(choices, size)) + '-' * (length - size)
            alt = '------'
    chromosome = np.random.choice(range(1, 25))
    position = np.random.sample()
    strand = np.random.choice([1, 2])
    return np.array(list(five_p)), np.array(list(three_p)), np.array(list(ref)), np.array(list(alt)), chromosome, position, strand


##to make sure the reverse of a simulated variant
def check_variant(variant, positive_variants):
    five_p = ''.join(variant[0])
    three_p = ''.join(variant[1])
    ref = ''.join(variant[2])
    alt = ''.join(variant[3])

    x = False
    for pos in positive_variants:
        if five_p == ''.join(pos[0]) and three_p == ''.join(pos[1]) and ref == ''.join(pos[2]) and alt == ''.join(pos[3]):
            x = True
            break
    if x:
        return x
    else:
        five_p_rev = str(Seq.reverse_complement(Seq(five_p)))
        three_p_rev = str(Seq.reverse_complement(Seq(three_p)))
        ref_rev = str(Seq.reverse_complement(Seq(ref.replace('-',''))))
        alt_rev = str(Seq.reverse_complement(Seq(alt.replace('-',''))))
        for pos in positive_variants:
            if five_p_rev == str(Seq.reverse_complement(Seq(''.join(pos[1])))) and three_p_rev == str(Seq.reverse_complement(Seq(''.join(pos[0])))) and ref_rev == str(Seq.reverse_complement(Seq(''.join(pos[2]).replace('-','')))) and alt_rev == str(Seq.reverse_complement(Seq(''.join(pos[3]).replace('-', '')))):
                x = True
                break
    return x

##how many different variants you want to label a positive sample
n_pos = 1
positive_choices = [generate_variant() for i in range(n_pos)]

def generate_sample(mean_variants=[50, 100, 300], mean_positive=.5, control=True, positive_choices=positive_choices):
    total_count = int(np.random.normal(np.random.choice(mean_variants, 1), 10))
    if total_count < 1:
        total_count *= -1
    if total_count == 0:
        total_count = 1
    if control:
        control_count = total_count
        positive_count = 0
    else:
        # positive_count = int(np.ceil(mean_positive * total_count))
        positive_count = 2
        control_count = total_count - positive_count
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
    positive_instances = []
    positive_variants = []
    for i in range(positive_count):
        positive_choice = np.random.choice(range(n_pos), 1)
        positive_variants.append(positive_choices[int(positive_choice)])
        positive_instances.append(int(positive_choice) + 1)
    return [control_variants + positive_variants, [0] * len(control_variants) + positive_instances]


##dictionary for instance level data
instances = {'sample_idx': [],
                 'seq_5p': [],
                 'seq_3p': [],
                  'seq_ref': [],
                  'seq_alt': [],
                  'gen_chr': [],
                  'gen_pos': [],
                  'strand': [],
                  'cds': [],
                  'class': []}

##
samples = {'classes': []}

for idx in range(500):
    if np.random.sample() < .5:
        variants = generate_sample()
        samples['classes'] = samples['classes'] + [0]
    else:
        variants = generate_sample(control=False, mean_positive=.02)
        samples['classes'] = samples['classes'] + [1]
    instances['sample_idx'] = instances['sample_idx'] + [idx] * len(variants[0])
    instances['seq_5p'] = instances['seq_5p'] + [i[0] for i in variants[0]]
    instances['seq_3p'] = instances['seq_3p'] + [i[1] for i in variants[0]]
    instances['seq_ref'] = instances['seq_ref'] + [i[2] for i in variants[0]]
    instances['seq_alt'] = instances['seq_alt'] + [i[3] for i in variants[0]]
    instances['gen_chr'] = instances['gen_chr'] + [i[4] for i in variants[0]]
    instances['gen_pos'] = instances['gen_pos'] + [i[5] for i in variants[0]]
    instances['strand'] = instances['strand'] + [i[6] for i in variants[0]]
    instances['cds'] = instances['cds'] + [0 for i in variants[0]]
    instances['class'] = instances['class'] + variants[1]

instances['sample_idx'] = np.array(instances['sample_idx'])
instances['seq_5p'] = np.array(instances['seq_5p'])
instances['seq_3p'] = np.array(instances['seq_3p'])
instances['seq_ref'] = np.array(instances['seq_ref'])
instances['seq_alt'] = np.array(instances['seq_alt'])
instances['chr'] = np.array(instances['gen_chr'])
instances['pos_float'] = np.array(instances['gen_pos'])
instances['strand'] = np.array(instances['strand'])


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

with open(cwd / 'sim_data' / 'sim_data.pkl', 'wb') as f:
    pickle.dump([instances, samples, ], f)



