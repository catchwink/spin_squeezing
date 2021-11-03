import numpy as np
import setup
from spin_dynamics import *
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()

    # only uniform all-to-all interactions in symmetry
    if interaction_shape == 'power_law':
        interaction_range_list = [0]
    else:
        interaction_range_list = [max(interaction_range_list)]

    method = 'XXZ'
    for interaction_range in interaction_range_list:
        spin_system = SpinOperators_Symmetry(system_size)
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')
        B_x = spin_system.get_Hamiltonian(['S_x'], [1.])
        B_y = spin_system.get_Hamiltonian(['S_y'], [1.])

        J_eff_list = [-0.1]

        total_T_vs_J_eff_vs_N = {}
        total_T_vs_J_eff_vs_N[10] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [2.003619125482685, 2.504523906853357, 3.3393652091378088, 5.009047813706714, 10.018095627413429, 20.036191254826857, 40.072382509653714, 80.14476501930743, 160.28953003861486, 160.28953003861486, 80.14476501930743, 40.072382509653714, 20.036191254826857]))
        total_T_vs_J_eff_vs_N[20] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [1.3416407864998738, 1.6770509831248424, 2.23606797749979, 3.3541019662496847, 6.708203932499369, 13.416407864998739, 26.832815729997478, 53.665631459994955, 107.33126291998991, 107.33126291998991, 53.665631459994955, 26.832815729997478, 13.416407864998739]))
        total_T_vs_J_eff_vs_N[50] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [0.7738576613305576, 0.967322076663197, 1.2897627688842628, 1.934644153326394, 3.869288306652788, 7.738576613305576, 15.477153226611152, 30.954306453222305, 61.90861290644461, 61.90861290644461, 30.954306453222305, 15.477153226611152, 7.738576613305576]))
        total_T_vs_J_eff_vs_N[100] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [0.5040000000000001, 0.6300000000000001, 0.8400000000000003, 1.2600000000000002, 2.5200000000000005, 5.040000000000001, 10.080000000000002, 20.160000000000004, 40.32000000000001, 40.32000000000001, 20.160000000000004, 10.080000000000002, 5.040000000000001]))
        total_T_vs_J_eff_vs_N[200] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [0.3247034339208626, 0.4058792924010783, 0.5411723898681045, 0.8117585848021566, 1.6235171696043131, 3.2470343392086263, 6.4940686784172525, 12.988137356834505, 25.97627471366901, 25.6, 12.8, 6.4, 3.2]))
        total_T_vs_J_eff_vs_N[500] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [0.16, 0.2, 0.26666666666666666, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 12.8, 6.4, 3.2, 1.6]))
        total_T_vs_J_eff_vs_N[1000] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [0.11535988904294245, 0.1441998613036781, 0.1922664817382375, 0.2883997226073562, 0.5767994452147124, 1.1535988904294248, 2.3071977808588495, 4.614395561717699, 9.228791123435398, 9.228791123435398, 4.614395561717699, 2.3071977808588495, 1.1535988904294248]))

        # J_eff_list = [-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01]
        for J_eff in [-0.1]:
            Jz = (J_eff + 1.) / 2
            Jperp = 2 * (1 - Jz)
            ham_terms_z = ['Sz_sq']
            ham_terms_perp = ['Sz_sq']
            strengths_z = [Jz]
            strengths_perp = [Jperp/2.]
            H_z = spin_system.get_Hamiltonian(ham_terms_z, strengths_z)
            H_perp = spin_system.get_Hamiltonian(ham_terms_perp, strengths_perp)
            spin_evolution = SpinEvolution((H_z, H_perp), psi_0, B=(B_x, B_y))
            total_T = total_T_vs_J_eff_vs_N[spin_system.N][J_eff]
            t_it = np.linspace(1., 1., 1)
            step_list = [1, 5, 10, 100, 1000, 5000, 10000]
            for steps in step_list:
                params = np.ones(3 * steps) * (total_T / steps)
                observed_t, t = spin_evolution.trotter_evolve_twice(params, t_it, observables=observables, store_states=False, discretize_time=True)
                observed_t['t'] = t
                util.store_observed_t(observed_t, 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_J_eff_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff, steps))