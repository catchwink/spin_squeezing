import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    # structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, coupling, instance = setup.configure(specify_coupling=True)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()
    
    # only uniform all-to-all interactions in symmetry
    if interaction_shape == 'power_law':
        interaction_range_list = [0]
    else:
        interaction_range_list = [max(interaction_range_list)]
        
    dirname = 'TFI_symm'

    variance_SN_vs_range_vs_method_h = {'TFI': {}}
    t_vs_range_vs_method_h = {'TFI': {}}
    method = 'TFI'
    spin_system = sd.SpinOperators_Symmetry(system_size)
    N = spin_system.N
    for interaction_range in interaction_range_list:
        # h_list = [coupling]
        Jz = -1
        h_list = N * Jz * np.array([-2., -1., -0.75, -0.5, -0.25, -0.125, -0.1, -0.05, -0.04, -0.03, -0.025, -0.02, -0.0125, -0.01, 0, 0.01, 0.0125, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.125, 0.25, 0.5, 0.75, 1., 2.])
        for h in h_list:
            filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range, Jz, h)
            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))

            variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
            t = observed_t['t']

            util.plot_variance_SN_vs_t(variance_SN_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname), coupling=h, coupling_name='h')
            util.plot_variance_norm_vs_t(variance_norm_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname), coupling=h, coupling_name='h')
            util.plot_angle_vs_t(angle_t, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='{}/plots'.format(dirname), coupling=h, coupling_name='h')
            
            variance_SN_vs_range_vs_method_h['TFI'][interaction_range] = variance_SN_t
            t_vs_range_vs_method_h['TFI'][interaction_range] = t

    util.plot_variance_SN_vs_t_all_ranges(variance_SN_vs_range_vs_method_h, t_vs_range_vs_method_h, N, interaction_shape, interaction_param_name, dirname='{}/plots'.format(dirname), coupling=h, coupling_name='h')
