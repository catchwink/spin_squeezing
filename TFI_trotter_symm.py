import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()

    # only uniform all-to-all interactions in symmetry
    if interaction_shape == 'power_law':
        interaction_range_list = [0]
    else:
        interaction_range_list = [max(interaction_range_list)]

    total_T_vs_N = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.165, 0.098, 0.047, 0.027000000000000003, 0.015, 0.007, 0.004]))

    method = 'TFI'
    for interaction_range in interaction_range_list:
        spin_system = sd.SpinOperators_Symmetry(system_size)
        N = spin_system.N
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')

        Jz = -1
        h = spin_system.N * Jz * 0.5
        H = spin_system.get_Hamiltonian(['Sz_sq'], [Jz])
        B = spin_system.get_Hamiltonian(['S_x'], [h])

        spin_evolution = sd.SpinEvolution(H, psi_0, B=B)

        t_it = np.linspace(1., 1., 1)
        step_list = [1, 5, 10, 100, 1000, 5000, 10000]
        for steps in step_list:
            total_T = total_T_vs_N[N]
            params = np.ones(2 * steps) * (total_T / steps)
            observed_t, t = spin_evolution.trotter_evolve_direct(params, t_it, observables=observables, store_states=False, discretize_time=True)
            observed_t['t'] = t
            util.store_observed_t(observed_t, 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps))

