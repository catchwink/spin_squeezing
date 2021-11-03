import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    # structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()

    # only uniform all-to-all interactions in symmetry
    if interaction_shape == 'power_law':
        interaction_range_list = [0]
    else:
        interaction_range_list = [max(interaction_range_list)]

    total_T_vs_N = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.20070140280561125, 0.13426853707414832, 0.07721442885771543, 0.050641282565130265, 0.03266533066132265, 0.01781563126252505, 0.01156312625250501]))
    min_variance_SN_vs_N = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.3098596100576169-3.3473972368523956e-18j), (0.19789688115378065-2.0871025016297664e-17j), (0.10404019502335474+1.1024533818862727e-17j), (0.0629522390791059+2.3697692662303067e-17j), (0.038004322048616625-1.1109637009729533e-17j), (0.019612998396892605+2.3944694324172194e-17j), (0.011950379725226045-1.6415165627961065e-17j)]))


    for system_size in [(10,1),(20,1),(50,1),(100,1),(200,1)]:
        spin_system = sd.SpinOperators_Symmetry(system_size)
        N = spin_system.N
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')

        H = spin_system.get_Hamiltonian(['Sz_sq'], [1.])
        B = spin_system.get_Hamiltonian(['S_y'], [1.])
        spin_evolution = sd.SpinEvolution(H, psi_0, B=B)

        total_T = total_T_vs_N[N]
        min_exact = min_variance_SN_vs_N[N]

        # version 1: at t_opt < 1, version 2: at t_opt < 2xmin, version 3: at t_min < 2xmin
        up_down_vs_N_vs_version = {}
        up_down_vs_N_vs_version[1] = dict(zip([10,20,50,100,200], [(10,100),(10,100),(100,1000),(100,1000),(100,1000)]))
        up_down_vs_N_vs_version[2] = dict(zip([10,20,50,100,200], [(10,100),(100,1000),(100,1000),(100,1000),(100,1000)]))
        up_down_vs_N_vs_version[3] = dict(zip([10,20,50,100,200], [(5,10),(10,100),(10,100),(10,100),(10,100)]))

        for version in [1,2,3]:
            down, up = up_down_vs_N_vs_version[version][N]
            method = 'ZX'
            for interaction_range in interaction_range_list:
                
                t_it = np.linspace(1., 1., 1)
                step_list = [int((up + down) / 2)]
                while len(step_list) > 0:
                    for steps in step_list:
                        params = np.ones(2 * steps) * (total_T / steps)
                        observed_t, t = spin_evolution.trotter_evolve(params, t_it, observables=observables, store_states=False, discretize_time=True)
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
