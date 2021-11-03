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

    total_T_vs_N = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.104, 0.0672, 0.036000000000000004, 0.0216, 0.012, 0.005600000000000001, 0.0032]))
    min_variance_SN_vs_N = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.27930329604706866-4.8871434199516e-17j), (0.16229489546823814+8.406013995130382e-17j), (0.0722320680944569-6.412969160985913e-17j), (0.03759725937879103+2.7255140213520144e-16j), (0.019384361968288702-7.563967980256765e-16j), (0.008027525988677058-2.670675983419243e-15j), (0.003950094500726353+2.233050141887525e-15j)]))

    for system_size in [system_size]:

        spin_system = sd.SpinOperators_Symmetry(system_size)
        N = spin_system.N
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')

        H_1 = spin_system.get_Hamiltonian(['Sz_sq'], [2.])
        H_2 = spin_system.get_Hamiltonian(['Sz_sq'], [1.])
        B = spin_system.get_Hamiltonian(['S_y'], [1.])
        spin_evolution = sd.SpinEvolution((H_1, H_2), psi_0, B=B)

        total_T = total_T_vs_N[N]
        min_exact = min_variance_SN_vs_N[N]

        # version 1: at t_opt < 1, version 2: at t_opt < 2xmin, version 3: at t_min < 2xmin
        up_down_vs_N_vs_version = {}
        up_down_vs_N_vs_version[1] = dict(zip([10,20,50,100,200], [(1,1),(1,5),(1,5),(1,5),(1,5),(5,10),(10,100)]))
        up_down_vs_N_vs_version[2] = dict(zip([10,20,50,100,200], [(1,5),(1,5),(5,10),(10,100),(10,100),(10,100),(100,1000)]))
        up_down_vs_N_vs_version[3] = dict(zip([10,20,50,100,200], [(1,5),(1,5),(1,5),(5,10),(10,100),(10,100),(10,100)]))

        for version in [1,2,3]:
            down, up = up_down_vs_N_vs_version[version][N]
            method = 'CTv2'
            for interaction_range in interaction_range_list:
                
                t_it = np.linspace(1., 1., 1)
                step_list = [int((up + down) / 2)]
                while len(step_list) > 0:
                    for steps in step_list:
                        params = np.ones(2 * steps) * (total_T / steps)
                        observed_t, t = spin_evolution.trotter_evolve_vary(params, t_it, observables=observables, store_states=False, discretize_time=True)
                        observed_t['t'] = t
                        util.store_observed_t(observed_t, 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps))
                        
                        min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(observed_t)
                        
                        if version == 1:
                            if min_variance_SN_t[-1] > 1 and up - down > 1:
                                down = steps
                                step_list = [int((up + down) / 2)]
                            elif min_variance_SN_t[-1] < 1 and up - down > 1:
                                up = steps
                                step_list = [int((up + down) / 2)]
                            else:
                                step_list = []

                        if version == 2:
                            if min_variance_SN_t[-1] > 2 * min_exact and up - down > 1:
                                down = steps
                                step_list = [int((up + down) / 2)]
                            elif min_variance_SN_t[-1] < 2 * min_exact and up - down > 1:
                                up = steps
                                step_list = [int((up + down) / 2)]
                            else:
                                step_list = []

                        if version == 3:
                            if min(min_variance_SN_t) > 2 * min_exact and up - down > 1:
                                down = steps
                                step_list = [int((up + down) / 2)]
                            elif min(min_variance_SN_t) < 2 * min_exact and up - down > 1:
                                up = steps
                                step_list = [int((up + down) / 2)]
                            else:
                                step_list = []
