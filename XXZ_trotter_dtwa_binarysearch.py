import numpy as np
import setup
from spin_dynamics import *
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()
    
    method = 'XXZ'
    spin_system = SpinOperators_DTWA(structure, system_size, fill)
    N = spin_system.N
    psi_0 = spin_system.get_init_state('x')
    B_x = spin_system.get_transverse_Hamiltonian('x')
    B_y = spin_system.get_transverse_Hamiltonian('y')

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

        J_eff_list = [-0.1, J_eff_plateau]
        for J_eff in J_eff_list:
            Jz = (J_eff + 1.) / 2
            Jperp = 2 * (1 - Jz)
            H_z = spin_system.get_Ising_Hamiltonian(Jz, interaction_range)
            H_perp = spin_system.get_Ising_Hamiltonian(Jperp/2., interaction_range)
            spin_evolution = SpinEvolution((H_z, H_perp), psi_0, B=(B_x, B_y))

            if N == 10 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 18.36
                min_exact = 0.31155078041228956
                up = 100
                down = 10
            elif N == 20 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 20.36
                min_exact = 0.2577957456296113
                up = 100
                down = 10
            elif N == 50 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 23.08
                min_exact = 0.22530096797476412
                up = 100
                down = 10
            elif N == 100 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 24.0
                min_exact = 0.20902731063529947
                up = 100
                down = 10
            elif N == 200 and interaction_range == 3 and J_eff == -0.1:
                total_T  = 24.48
                min_exact = 0.2030014132416131
                up = 100
                down = 10

            if N == 10 and interaction_range == 3 and J_eff == -0.06:
                total_T  = 31.07
                min_exact = 0.28398275586179905
                up = 1000
                down = 100
            elif N == 20 and interaction_range == 3 and J_eff == -0.02:
                total_T  = 115.60
                min_exact = 0.19147429935615187
                up = 2000
                down = 1000
            elif N == 50 and interaction_range == 3 and J_eff == -0.005:
                total_T  = 664.80
                min_exact = 0.10268304955463758
                up = 500
                down = 100
            elif N == 100 and interaction_range == 3 and J_eff == -0.005:
                total_T  = 800.00
                min_exact = 0.06896338844060595
                up = 200
                down = 100
            elif N == 200 and interaction_range == 3 and J_eff == -0.0025:
                total_T  = 1600.00
                min_exact = 0.051241270965356876
                up = 200
                down = 100

            t_it = np.linspace(1., 1., 1)
            step_list = [int((up + down) / 2)]
            while len(alphas) > 0:
                for steps in step_list:
                    params = np.ones(3 * steps) * (total_T / steps)
                    tdist = spin_evolution.trotter_evolve_twice(params, t_it, store_states=True, discretize_time=True)
                    meanConfig_evol = np.mean(tdist,axis=1)

                    min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
                    results_t = spin_system.get_observed(tdist, meanConfig_evol)
                    results_t['min_variance_SN'] = min_variance_SN_t
                    results_t['min_variance_norm'] = min_variance_norm_t
                    results_t['opt_angle'] = opt_angle_t

                    util.store_observed_t(results_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff, steps))

                    # if min_variance_SN_t[-1] > 1 and up - down > 1:
                    # if min_variance_SN_t[-1] > 2 * min_exact and up - down > 1:
                    if min(min_variance_SN_t) > 2 * min_exact and up - down > 1:
                        down = alpha
                        step_list = [int((up + down) / 2)]
                    # elif min_variance_SN_t[-1] < 1 and up - down > 1:
                    # elif min_variance_SN_t[-1] < 2 * min_exact and up - down > 1:
                    elif min(min_variance_SN_t) < 2 * min_exact and up - down > 1:
                        up = alpha
                        step_list = [int((up + down) / 2)]
                    else:
                        step_list = []
