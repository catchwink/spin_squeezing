import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()

    method = 'ZX'
    for interaction_range in interaction_range_list[-2:]:
        print(interaction_range)
        spin_system = sd.SpinOperators_QuSpin(structure, system_size, fill, interaction_shape, interaction_range)
        N = spin_system.N
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')
        H = spin_system.get_Hamiltonian(['zz'])
        B = spin_system.get_Hamiltonian(['y'])
        spin_evolution = sd.SpinEvolution(H, psi_0, B=B)
        
        if N == 4:
            total_T = 0.343
        elif N == 8:
            total_T  = 0.228
        elif N == 16:
            total_T  = 0.153
        elif N == 32:
            total_T  = 0.101
        elif N == 64:
            total_T  = 0.066
        elif N == 128:
            total_T  = 0.043
        elif N == 256:
            total_T  = 0.028
        elif N == 512:
            total_T  = 0.018
        elif N == 1024:
            total_T  = 0.011
        elif N == 2048:
            total_T  = 0.007
        elif N == 4096:
            total_T  = 0.005

        t_it = np.linspace(1., 1., 1)
        step_list = [1, 5, 10, 100, 1000, 5000, 10000]
        for steps in step_list:
            params = np.ones(2 * steps) * (total_T / steps)
            observed_t = spin_evolution.trotter_evolve(params, t_it, observables=observables, store_states=False, discretize_time=True)
            util.store_observed_t(observed_t, 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps))