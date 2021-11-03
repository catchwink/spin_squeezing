import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, instance = setup.configure(specify_range=True)
    
    method = 'XXZ'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    N = spin_system.N
    psi_0 = spin_system.get_init_state('x')
    B_x = spin_system.get_transverse_Hamiltonian('x')
    B_y = spin_system.get_transverse_Hamiltonian('y')

    interaction_range_list = [interaction_range]
    for interaction_range in interaction_range_list:

        if N == 10 and interaction_range == 3:
            J_eff_plateau = -0.06
        elif N == 20 and interaction_range == 3:
            J_eff_plateau = -0.02
        elif N == 50 and interaction_range == 3:
            J_eff_plateau = -0.005
        elif N == 100 and interaction_range == 3:
            J_eff_plateau = -0.005
        elif N == 200 and interaction_range == 3:
            J_eff_plateau = -0.0025

        # J_eff_list = [-0.1, J_eff_plateau]
        J_eff_list = [-0.1]
        for J_eff in J_eff_list:
            Jz = (J_eff + 1.) / 2
            Jperp = 2 * (1 - Jz)
            H_z = spin_system.get_Ising_Hamiltonian(Jz, interaction_range)
            H_perp = spin_system.get_Ising_Hamiltonian(Jperp/2., interaction_range)
            spin_evolution = sd.SpinEvolution((H_z, H_perp), psi_0, B=(B_x, B_y))

            total_T = 5.
            if N == 10 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 18.36
            elif N == 20 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 20.36
            elif N == 50 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 23.08
            elif N == 100 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 24.0
            elif N == 200 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 24.48

            if N == 10 and interaction_range == 3 and J_eff == -0.06:
                total_T  = 31.07
            elif N == 20 and interaction_range == 3 and J_eff == -0.02:
                total_T  = 115.60
            elif N == 50 and interaction_range == 3 and J_eff == -0.005:
                total_T  = 664.80
            elif N == 100 and interaction_range == 3 and J_eff == -0.005:
                total_T  = 800.00
            elif N == 200 and interaction_range == 3 and J_eff == -0.0025:
                total_T  = 1600.00

            t_it = np.linspace(1., 1., 1)
            step_list = [1, 5, 10, 100, 1000, 5000, 10000]
            for steps in step_list:
                params = np.ones(3 * steps) * (total_T / steps)
                tdist, t = spin_evolution.trotter_evolve_twice(params, t_it, store_states=True, discretize_time=True)
                meanConfig_evol = np.mean(tdist,axis=1)

                min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
                results_t = spin_system.get_observed(tdist, meanConfig_evol)
                results_t['min_variance_SN'] = min_variance_SN_t
                results_t['min_variance_norm'] = min_variance_norm_t
                results_t['opt_angle'] = opt_angle_t
                results_t['t'] = t

                util.store_observed_t(results_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff, steps))

                