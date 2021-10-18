import numpy as np
import pylab as plt
from clonesig.data_loader import SimLoader
# from clonesig.run_clonesig import get_MU, run_clonesig
import pickle
import pathlib
path = pathlib.Path.cwd()


if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))


tables = []
purities = []
mean_variants = [300, 500, 1000, 1500, 2000, 2500, 3000]
for i in range(10000):
    if i % 10 == 0:
        print(i)
    center = np.random.choice(mean_variants, 1)
    N = int(np.random.normal(center, int(np.ceil(center * .2))))
    assert N > 0
    J = np.random.choice([1, 2, 3, 4])
    steady_phi = np.zeros(J)
    steady_phi[0] = 1.0
    # get steady_phi in decreasing order
    for i in range(1, J):
        steady_phi[i] = np.random.uniform(
            low=0.1 + 0.1 * (J - i - 1),
            high=steady_phi[i-1] - 0.1)

    steady_xi = np.random.dirichlet(alpha=np.ones(J))
    while min(steady_xi) < 0.1:
        steady_xi = np.random.dirichlet(alpha=np.ones(J))

    sim_object = SimLoader(N,
                           J,
                           rho_param=100,
                           cn=False,
                           xi_param=steady_xi,
                           phi_param=steady_phi
                           )
    sim_object._get_unobserved_nodes()
    sim_object._get_observed_nodes()
    # to get the mutation table
    sim_mutation_table = sim_object._get_data_df()
    tables.append(sim_mutation_table)
    purities.append(sim_object.purity)

with open(cwd / 'figures' / 'vaf' / 'sim_data.pkl', 'wb') as f:
    pickle.dump([tables, purities], f)