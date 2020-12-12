from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import pylab as plt
import pandas as pd
from scipy.stats import percentileofscore
from sim_data.sim_data_tools import *
import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'ATGC2':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC2')]


def generate_times(n=200, mean_time=365, risk=0):
    risk_score = np.full((n), risk)
    baseline_hazard = 1 / mean_time
    scale = baseline_hazard * np.exp(risk_score)
    u = np.random.uniform(low=0, high=1, size=len(risk_score))
    t = -np.log(u) / scale
    low_qt = np.quantile(t, .05)
    high_qt = np.quantile(t, .9)
    c = np.random.uniform(low=low_qt, high=high_qt, size=n)
    c *= np.array([np.random.choice([0.7, 1], p=[percent, 1-percent]) for percent in np.array([percentileofscore(t, i) for i in t]) / 100])
    observed_event = t <= c
    observed_time = np.where(observed_event, t, c)
    return observed_time, observed_event



def generate_sample(mean_variants=[5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300],
                    mean_positive=None, num_positive=None, control=True, positive_choices=None, negative_instances=False):
    if negative_instances and len(positive_choices) <= 1:
        raise ValueError
    center = np.random.choice(mean_variants, 1)
    total_count = int(np.random.normal(center, int(np.ceil(center * .2))))
    if total_count < 1:
        total_count *= -1
    if total_count == 0:
        total_count = np.random.choice([2, 3, 4, 5, 6], 1)
    if control:
        if negative_instances:
            if num_positive:
                positive_count = num_positive
            else:
                positive_count = int(np.ceil(mean_positive * total_count))
            control_count = total_count - positive_count
        else:
            control_count = total_count
            positive_count = 0
    else:
        if num_positive:
            positive_count = num_positive
        else:
            positive_count = int(np.ceil(mean_positive * total_count))
        control_count = total_count - positive_count * len(positive_choices)

    control_count = max(control_count, 0)
    positive_variants = []
    positive_instances = []

    control_variants = [generate_variant() for i in range(control_count)]
    if control:
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

    if control:
        if negative_instances:
            positive_choice = int(np.random.choice(range(len(positive_choices)), 1))
            for i in range(positive_count):
                positive_variants.append(positive_choices[positive_choice])
                positive_instances.append(positive_choice + 1)
        else:
            pass

    else:
        for index, i in enumerate(positive_choices):
            for ii in range(positive_count):
                positive_variants.append(i)
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
        variants = generate_sample(control=False, mean_positive=.1, positive_choices=positive_choices)
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

samples['classes'] = np.array(samples['classes'])

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

##generate times
control_data = generate_times(n=sum(samples['classes'] == 0), risk=0)
positive_data = generate_times(n=sum(samples['classes'] == 1), risk=2)

samples['times'] = []
samples['event'] = []
control_count = 0
positive_count = 0
for i in samples['classes']:
    if i == 0:
        samples['times'].append(control_data[0][control_count])
        samples['event'].append(control_data[1][control_count])
        control_count += 1
    else:
        samples['times'].append(positive_data[0][positive_count])
        samples['event'].append(positive_data[1][positive_count])
        positive_count += 1

samples['times'] = np.array(samples['times'])
samples['event'] = np.array(samples['event'])


##plotting
# fig=plt.figure()
# ax = fig.add_subplot(111)
#
# kmf_low = KaplanMeierFitter()
# kmf_low.fit(control_data[0], control_data[1])
# # kmf_low.survival_function_.plot()
# kmf_low.plot(show_censors=True, ci_show=False, ax=ax)
# #
# kmf_high = KaplanMeierFitter()
# kmf_high.fit(positive_data[0], positive_data[1])
# # # kmf_high.survival_function_.plot()
# kmf_high.plot(show_censors=True, ci_show=False, ax=ax)
# #
# plt.show()
#

# ##lifelines
concordance_index(samples['times'], np.exp(-1 * samples['classes']), samples['event'])

with open(cwd / 'sim_data' / 'survival' / 'experiment_1' / 'sim_data.pkl', 'wb') as f:
    pickle.dump([instances, samples, ], f)

##cox regression

##lifelines
##need a dataframe

# d = {'risks': np.concatenate([np.zeros(500), np.ones(500)]),
#      'times': np.concatenate([low_risk_group[0], high_risk_group[0]]),
#      'events': np.concatenate([low_risk_group[1], high_risk_group[1]])}
# data = pd.DataFrame(data=d)
#
#
# cph = CoxPHFitter()
# cph.fit(data, 'times', 'events')
# cph.print_summary()






