import numpy as np
import setup
import spin_dynamics as sd
import os
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()
    
    method = 'CTv2'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    psi_0 = spin_system.get_init_state('x')
    for interaction_range in interaction_range_list:
        for J in [1.]:
            H = spin_system.get_CTv2_Hamiltonian(J, interaction_range)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            t_max = 4.
            t = np.linspace(t_max/500, t_max, 500)
            tdist, t = spin_evolution.evolve([1.], t, store_states=True)
            print(tdist.shape)

            result_dir = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J)
            os.mkdir(result_dir)
            for bs in range(50):
                samples = np.random.choice(10000, size=10000)
                tdist_sample = tdist[:,samples]
                print(tdist_sample.shape)
                meanConfig_evol_sample = np.mean(tdist_sample,axis=1)
                min_variance_SN_t_sample, min_variance_norm_t_sample, opt_angle_t_sample = spin_system.get_squeezing(tdist_sample, meanConfig_evol_sample)
        
                results_t_sample = spin_system.get_observed(tdist_sample, meanConfig_evol_sample)
                results_t_sample['min_variance_SN'] = min_variance_SN_t_sample
                results_t_sample['min_variance_norm'] = min_variance_norm_t_sample
                results_t_sample['opt_angle'] = opt_angle_t_sample
                results_t_sample['t'] = t

                util.store_observed_t(results_t_sample, '{}/bootstrap_{}'.format(result_dir, bs))