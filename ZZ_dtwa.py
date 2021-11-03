import numpy as np
import setup
import spin_dynamics as sd
import util
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, instance = setup.configure(specify_range=True)
    
    method = 'ZZ'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    psi_0 = spin_system.get_init_state('x')
    for interaction_range in [interaction_range]:
        J = 1.
        H = spin_system.get_Ising_Hamiltonian(J, interaction_range)
        spin_evolution = sd.SpinEvolution(H, psi_0)
        n_steps = 100 if (spin_system.N < 1000 or interaction_range > 0) else 50
        t_max = 1.5
            
        t = np.linspace(t_max/n_steps, t_max, n_steps)
        tdist, t = spin_evolution.evolve([1.], t, store_states=True)
        meanConfig_evol = np.mean(tdist,axis=1)
        min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
        
        idx_first_min_variance_SN = argrelextrema(min_variance_SN_t, np.less)[0][0]
        t_opt_coarse = t[idx_first_min_variance_SN]

        n_steps_fine = 2 * (int(t_max / t_opt_coarse) + 1)
        t_fine = np.linspace(t[idx_first_min_variance_SN - 1], t[idx_first_min_variance_SN + 1], n_steps_fine + 1)
            
        t = np.sort(np.concatenate((t[1:], t_fine)))
        tdist, t = spin_evolution.evolve([1.], t, store_states=True)
        meanConfig_evol = np.mean(tdist,axis=1)
        min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)

        results_t = spin_system.get_observed(tdist, meanConfig_evol)
        results_t['min_variance_SN'] = min_variance_SN_t
        results_t['min_variance_norm'] = min_variance_norm_t
        results_t['opt_angle'] = opt_angle_t
        results_t['t'] = t

        util.store_observed_t(results_t, 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range))