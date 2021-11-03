import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()
    
    method = 'XXZ'
    interaction_range_list = [3.]
    for interaction_range in interaction_range_list:
        spin_system = sd.SpinOperators_QuSpin(structure, system_size, fill, interaction_shape, interaction_range)
        N = spin_system.N
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')
        H_var = spin_system.get_var_Hamiltonian({'S_z^2':['zz']})
        B_x = spin_system.get_Hamiltonian(['x'])
        B_y = spin_system.get_Hamiltonian(['y'])

        J_eff_list = [-0.1] ### FILL IN
        for J_eff in J_eff_list:
            Jz = (J_eff + 1.) / 2
            Jperp = 2 * (1 - Jz)
            params_dict_z = {'S_z^2':Jz}
            params_dict_perp = {'S_z^2':(Jperp/2.)}
            H_z = H_var.tohamiltonian(params_dict_z)
            H_perp = H_var.tohamiltonian(params_dict_perp)
            spin_evolution = sd.SpinEvolution((H_z, H_perp), psi_0, B=(B_x, B_y))

            total_T = 18.36 ### ALTER THIS
            t_it = np.linspace(1., 1., 1)
            step_list = [1, 5, 10, 100, 1000, 5000, 10000]
            for steps in step_list:
                params = np.ones(3 * steps) * (total_T / steps)
                observed_t = spin_evolution.trotter_evolve_twice(params, t_it, observables=observables, store_states=False, discretize_time=True)
                util.store_observed_t(observed_t, 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_J_eff_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff, steps))