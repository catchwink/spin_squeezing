import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()

    for interaction_range in interaction_range_list:
        spin_system = sd.SpinOperators_QuSpin(structure, system_size, fill, interaction_shape, interaction_range)
        N = spin_system.N
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')
        for method in ['ZZ', 'XY']:
            if method == 'ZZ':
                ham_terms = ['zz']
            elif method == 'XY':
                ham_terms = ['xx', 'yy']
            H = spin_system.get_Hamiltonian(ham_terms)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            t = np.linspace(.01, 5., 500)
            observed_t = spin_evolution.evolve([1.], t, observables=observables, store_states=False)
            util.store_observed_t(observed_t, 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range))