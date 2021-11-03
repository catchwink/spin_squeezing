import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, instance = setup.configure(specify_range=True)

    method = 'CTv2'
    for interaction_range in [interaction_range]:
        spin_system = sd.SpinOperators_QuSpin(structure, system_size, fill, interaction_shape, interaction_range)
        N = spin_system.N
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')
        H_var = spin_system.get_var_Hamiltonian({'S_z^2':['zz'], 'S_x^2':['xx']})
        for J in [1.]:
            params_dict = {'S_z^2':(2 * J), 'S_x^2':J}
            H = H_var.tohamiltonian(params_dict)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            t = np.linspace(.01, 5., 500)
            observed_t, t = spin_evolution.evolve([1.], t, observables=observables, store_states=False)
            observed_t['t'] = t
            filename_N = '{}'.format(spin_system.N) if (system_size[0] == 1 or system_size[1] == 1) else '{}_{}'.format(system_size[0], system_size[1])
            util.store_observed_t(observed_t, 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, filename_N, interaction_shape, interaction_param_name, interaction_range))