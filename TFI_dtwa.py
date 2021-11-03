import numpy as np
import setup
import spin_dynamics as sd
import util
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, field, instance = setup.configure(specify_range=True, specify_coupling=True)
    
    method = 'TFI'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    N = spin_system.N
    psi_0 = spin_system.get_init_state('x')
    for interaction_range in [interaction_range]:
        for Jz in [-1.]:
            # h_list = np.concatenate(([spin_system.N * Jz * 0.5], np.arange(-5,5.5,0.5)))
            # h_list = spin_system.N * Jz * np.arange(-1,1.25,0.25)
            for h in [field, spin_system.N * Jz * field]:
                H = spin_system.get_TFI_Hamiltonian(Jz, h, interaction_range)
                spin_evolution = sd.SpinEvolution(H, psi_0)
                n_steps = 100
                t_max = 1.
                
                t = np.linspace(t_max/n_steps, t_max, n_steps)
                tdist, t = spin_evolution.evolve([1.], t, store_states=True)
                meanConfig_evol = np.mean(tdist,axis=1)
                min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)

                idx_first_min_variance_SN = argrelextrema(min_variance_SN_t, np.less)[0][0]
                print(argrelextrema(min_variance_SN_t, np.less), idx_first_min_variance_SN)
                t_opt_coarse = t[idx_first_min_variance_SN]
                print(t_max, t_opt_coarse)

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

                util.store_observed_t(results_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h))