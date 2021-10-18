import numpy as np
from clonesig.run_clonesig import get_MU, run_clonesig

import pickle
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

default_MU = get_MU()
print('size of the default MU matrix:', default_MU.shape)

predictions = []
for idx in idx_test:
    sim_mutation_table = tables[idx]
    print('true_clones', len(sim_mutation_table['clone'].unique()))
    purity = purities[idx]
    est, lr, pval, new_inputMU, cst_est, future_sigs = run_clonesig(
        np.array(sim_mutation_table.trinucleotide),
        np.array(sim_mutation_table.var_counts),
        np.array(sim_mutation_table.var_counts + sim_mutation_table.ref_counts),
        np.array(sim_mutation_table.normal_cn),
        np.array(sim_mutation_table.total_cn),
        np.array(sim_mutation_table.total_cn - sim_mutation_table.major_cn),
        purity, default_MU)
    predictions.append(est.J)

with open(cwd / 'figures' / 'vaf' / 'clonesig_predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)