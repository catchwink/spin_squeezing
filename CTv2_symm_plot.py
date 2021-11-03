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

    dirname = 'CTv2_symm'

    variance_SN_vs_range_vs_method_h = {'CTv2': {}}
    t_vs_range_vs_method_h = {'CTv2': {}}
    method = 'CTv2'
    spin_system = sd.SpinOperators_Symmetry(system_size)
    N = spin_system.N
    for interaction_range in interaction_range_list:
        for J in [1.]:
            filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J)
            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
            variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
            t = observed_t['t']

            util.plot_variance_SN_vs_t(variance_SN_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname))
            util.plot_variance_norm_vs_t(variance_norm_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname))
            util.plot_angle_vs_t(angle_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname))
            
            variance_SN_vs_range_vs_method_h['CTv2'][interaction_range] = variance_SN_t
            t_vs_range_vs_method_h['CTv2'][interaction_range] = t

    util.plot_variance_SN_vs_t_all_ranges(variance_SN_vs_range_vs_method_h, t_vs_range_vs_method_h, N, interaction_shape, interaction_param_name, dirname='{}/plots'.format(dirname))
