import numpy as np
import setup
import spin_dynamics as sd
import os
import util
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, n_trotter_steps, instance = setup.configure(specify_range=True, specify_trotter=True)

    method = 'ZX'
    structure = 'inhomogeneous'
    fill = 'N/A'
    N = system_size[0] * system_size[1]
    interaction_shape = 'RD'
    interaction_param_name = 'N/A'
    interaction_range = 'N/A'

    with open('../inhomogeneous_dtwa/inhomogeneous_instance_N_{}.npy'.format(N), 'rb') as f:
        coord = np.load(f)
        interaction_matrix = np.load(f)
    print(method, structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, instance)
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill, coord=coord)

    psi_0 = spin_system.get_init_state('x')
    B = spin_system.get_transverse_Hamiltonian('y')
    for interaction_range in [interaction_range]:
        for J in [1.]:
            H = spin_system.get_Ising_Hamiltonian(J, [], Jij=interaction_matrix)
            spin_evolution = sd.SpinEvolution(H, psi_0, B=B)

            cont_filename = 'observables_vs_t_{}_{}_N_{}_{}_J_{}'.format('XY', structure, N, interaction_shape, J)
            cont_dirname = '../{}_dtwa'.format(structure)
            if cont_filename in os.listdir(cont_dirname):
                cont_observed_t = util.read_observed_t('{}/{}'.format(cont_dirname, cont_filename))
                cont_variance_SN_t, cont_variance_norm_t, cont_angle_t, cont_t = cont_observed_t['min_variance_SN'], cont_observed_t['min_variance_norm'], cont_observed_t['opt_angle'], cont_observed_t['t']
                
                idx_first_cont_variance_SN = argrelextrema(cont_variance_SN_t, np.less)[0][0]
                total_T = cont_t[idx_first_cont_variance_SN]

                t_it = np.linspace(1., 1., 1)
                for steps in [n_trotter_steps]:
                    params = np.ones(2 * steps) * (total_T / steps)
                    tdist, t = spin_evolution.trotter_evolve(params, t_it, store_states=True, discretize_time=True)
                    meanConfig_evol = np.mean(tdist,axis=1)

                    min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
                    results_t = spin_system.get_observed(tdist, meanConfig_evol)
                    results_t['min_variance_SN'] = min_variance_SN_t
                    results_t['min_variance_norm'] = min_variance_norm_t
                    results_t['opt_angle'] = opt_angle_t
                    results_t['t'] = t

                    util.store_observed_t(results_t, 'observables_vs_t_{}_{}_N_{}_{}_J_{}_steps_{}'.format(method, structure, N, interaction_shape, J, steps))
