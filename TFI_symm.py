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

    method = 'TFI'
    spin_system = sd.SpinOperators_Symmetry(system_size)
    observables = spin_system.get_observables()
    psi_0 = spin_system.get_init_state('x')
    for interaction_range in interaction_range_list:
        Jz = -1
        # h_list1 = spin_system.N * Jz * np.array([-2., -1., -0.75, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 0.75, 1., 2.])
        # h_list2 = spin_system.N * Jz * np.array([-0.1, -0.05, -0.025, -0.0125, 0.0125, 0.025, 0.05, 0.1])
        # h_list3 = spin_system.N * Jz * np.array([-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04])
        # h_list4 = np.arange(-5,5.5,0.5)
        # for h in np.concatenate([h_list1, h_list2, h_list3, h_list4]):
        for h in np.concatenate((np.arange(5.5, 10.5, 0.5), -np.arange(5.5, 10.5, 0.5))):
            ham_terms = ['Sz_sq', 'S_x']
            strengths = [Jz, h]
            H = spin_system.get_Hamiltonian(ham_terms, strengths)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            # t_max = 4.
            t_max = 0.5
            t = np.linspace(t_max/500, t_max, 500)
            observed_t, t = spin_evolution.evolve([1.], t, observables=observables, store_states=False)
            observed_t['t'] = t
            util.store_observed_t(observed_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h))