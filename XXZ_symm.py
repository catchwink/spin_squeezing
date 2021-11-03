import numpy as np
import setup
import spin_dynamics as sd
import util
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, coupling, instance = setup.configure(specify_coupling=True)

    # only uniform all-to-all interactions in symmetry
    if interaction_shape == 'power_law':
        interaction_range_list = [0]
    else:
        interaction_range_list = [max(interaction_range_list)]

    method = 'XXZ'
    for interaction_range in interaction_range_list:
        spin_system = sd.SpinOperators_Symmetry(system_size)
        observables = spin_system.get_observables()
        psi_0 = spin_system.get_init_state('x')
        for J_eff in [coupling]:
            Jz = (J_eff + 1.) / 2
            Jperp = 2 * (1 - Jz)
            ham_terms = ['Sz_sq', 'Sx_sq', 'Sy_sq']
            strengths = [Jz, Jperp/2., Jperp/2.]
            H = spin_system.get_Hamiltonian(ham_terms, strengths)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            # t_max = 4. / np.abs(J_eff)
            t_max = 0.8 * np.power(spin_system.N, -0.5) / np.abs(J_eff)
            t = np.linspace(t_max/500, t_max, 500)
            observed_t, t = spin_evolution.evolve([1.], t, observables=observables, store_states=False)
            observed_t['t'] = t
            util.store_observed_t(observed_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff))