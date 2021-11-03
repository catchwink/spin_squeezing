import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()
    
    dirname = 'ZZ_dtwa'

    variance_SN_vs_range_vs_method_J_eff_1 = {'ZZ': {}}
    t_vs_range_vs_method_J_eff_1 = {'ZZ': {}}
    method = 'ZZ'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    N = spin_system.N
    for interaction_range in interaction_range_list:
        for J_eff in [1.0]:
            filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))

            variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']

            util.plot_variance_SN_vs_t(variance_SN_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname), coupling=J_eff)
            util.plot_variance_norm_vs_t(variance_norm_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname), coupling=J_eff)
            util.plot_angle_vs_t(angle_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname), coupling=1.)
            
            if J_eff == 1.:
                variance_SN_vs_range_vs_method_J_eff_1['ZZ'][interaction_range] = variance_SN_t
                t_vs_range_vs_method_J_eff_1['ZZ'][interaction_range] = t

    util.plot_variance_SN_vs_t_all_ranges(variance_SN_vs_range_vs_method_J_eff_1, t_vs_range_vs_method_J_eff_1, N, interaction_shape, interaction_param_name, dirname='{}/plots'.format(dirname), coupling=1.0)
