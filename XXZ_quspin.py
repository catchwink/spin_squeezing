import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()
    
    method = 'XXZ'
    for interaction_range in interaction_range_list:
        spin_system = sd.SpinOperators_QuSpin(structure, system_size, fill, interaction_shape, interaction_range)
        N = spin_system.N
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')
        H_var = spin_system.get_var_Hamiltonian({'S_z^2':['zz'], 'S_x^2 + S_y^2':['xx', 'yy']})

        J_eff_list = [-0.1] ### FILL IN
        for J_eff in J_eff_list:
            Jz = (J_eff + 1.) / 2
            Jperp = 2 * (1 - Jz)
            params_dict = {'S_z^2':Jz, 'S_x^2 + S_y^2':(Jperp/2.)}
            H = H_var.tohamiltonian(params_dict)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            t_max = 4. / np.abs(J_eff)
            t = np.linspace(t_max/500, t_max, 500)
            observed_t = spin_evolution.evolve([1.], t, observables=observables, store_states=False)
            util.store_observed_t(observed_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff))