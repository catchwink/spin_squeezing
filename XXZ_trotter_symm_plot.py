import numpy as np
import setup
import spin_dynamics as sd
import util

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, instance = setup.configure(specify_range=True)
    
    # only uniform all-to-all interactions in symmetry
    if interaction_shape == 'power_law':
        interaction_range_list = [0]
    else:
        interaction_range_list = [max(interaction_range_list)]

    dirname = 'XXZ_trotter_symm'

    method = 'XXZ'
    spin_system = sd.SpinOperators_Symmetry(system_size)
    N = spin_system.N
    
    variance_SN_vs_delta_t_vs_N = {N: {}}
    for interaction_range in interaction_range_list:
        J_eff_list = [-0.1]
        for J_eff in J_eff_list:
            step_list = [1, 5, 10, 100, 1000, 5000, 10000]
            variance_SN_t_vs_steps = {}
            variance_norm_t_vs_steps = {}
            angle_t_vs_steps = {}
            t_vs_steps = {}
            components_vs_steps = {}
            for steps in step_list:
                filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_J_eff_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff, steps)
                observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                t = observed_t['t']
                components_t = (observed_t['S_x'], observed_t['S_y'], observed_t['S_z'])

                variance_SN_t_vs_steps[steps] = variance_SN_t
                variance_norm_t_vs_steps[steps] = variance_norm_t
                angle_t_vs_steps[steps] = angle_t
                t_vs_steps[steps] = t
                components_vs_steps[steps] = components_t
                total_T = t[-1] / 3.
                variance_SN_vs_delta_t_vs_N[N][(J_eff, total_T / steps)] = variance_SN_t[-1]

            util.plot_variance_SN_vs_t_trotter(variance_SN_t_vs_steps, t_vs_steps, total_T, method, N, interaction_shape, interaction_param_name, interaction_range, dirname=dirname + '/plots', J_eff=J_eff)
            util.plot_variance_norm_vs_t_trotter(variance_norm_t_vs_steps, t_vs_steps, total_T, method, N, interaction_shape, interaction_param_name, interaction_range, dirname=dirname + '/plots', J_eff=J_eff)
            
            util.plot_components_vs_t_trotter(components_vs_steps, t_vs_steps, total_T, method, N, interaction_shape, interaction_param_name, interaction_range, dirname=dirname + '/plots', J_eff=J_eff)
            util.plot_signal_noise_vs_t_trotter(components_vs_steps, variance_norm_t_vs_steps, t_vs_steps, total_T, method, N, interaction_shape, interaction_param_name, interaction_range, dirname=dirname + '/plots', J_eff=J_eff)
            
    util.plot_variance_SN_vs_delta_t_trotter(variance_SN_vs_delta_t_vs_N, method, interaction_shape, interaction_param_name, interaction_range, dirname=dirname + '/plots', J_eff=J_eff)
