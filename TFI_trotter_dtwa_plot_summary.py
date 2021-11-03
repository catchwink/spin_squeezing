import numpy as np
import setup
import spin_dynamics as sd
import util
from matplotlib import pyplot as plt
import os
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    method = 'TFI'
    dirname = 'TFI_trotter_dtwa'
    interaction_shape = 'power_law'
    interaction_param_name = 'exp'

    total_T_vs_N_vs_range = {}
    for interaction_range in [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.]:

        # version 1: at t_opt < 1, version 2: at t_opt < 2xmin, version 3: at t_min < 2xmin
        step_upper_vs_N_vs_version = {1:{},2:{},3:{}}
        step_lower_vs_N_vs_version = {1:{},2:{},3:{}}

        total_T_vs_N = {}
        min_variance_SN_cont_vs_N = {}
        for N in [10, 20, 50, 100, 200, 500, 1000]:
            Jz = -1.
            h = N * Jz / 2
            cont_filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method, N, 'power_law', 'exp', interaction_range, Jz, h)
            cont_dirname = '{}_dtwa'.format(method)
            if cont_filename in os.listdir(cont_dirname):
                cont_observed_t = util.read_observed_t('{}/{}'.format(cont_dirname, cont_filename))
                cont_variance_SN_t, cont_variance_norm_t, cont_angle_t, cont_t = cont_observed_t['min_variance_SN'], cont_observed_t['min_variance_norm'], cont_observed_t['opt_angle'], cont_observed_t['t']
                
                if len(argrelextrema(cont_variance_SN_t, np.less)[0]) > 0:
                    idx_first_cont_variance_SN = argrelextrema(cont_variance_SN_t, np.less)[0][0]
                    total_T = cont_t[idx_first_cont_variance_SN]
                    total_T_vs_N[N] = total_T

                    min_variance_SN_cont = cont_variance_SN_t[idx_first_cont_variance_SN]
                    min_variance_SN_cont_vs_N[N] = min_variance_SN_cont

                    system_size = (N, 1)
                    spin_system = sd.SpinOperators_Symmetry(system_size)

                    fig = plt.figure()
                    plt.title(r'TFI, $J_z = -1$, $h = N \cdot J_z /2$, $N = {}$, power_law, $\alpha = {}$'.format(N, interaction_range))
                    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                    plt.xlabel('# steps')
                    # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                    variance_SN_t_min = []
                    variance_SN_t_opt = []
                    # for steps in step_list:
                    step_list = []
                    print(N, interaction_range)
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h, steps)
                        if filename in os.listdir(dirname):
                            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                            variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                            t = observed_t['t']
                            variance_SN_t_opt.append(variance_SN_t[-1])
                            variance_SN_t_min.append(np.min(variance_SN_t))
                            step_list.append(steps)
                    step_list = np.array(step_list)
                    variance_SN_t_opt = np.array(variance_SN_t_opt)
                    variance_SN_t_min = np.array(variance_SN_t_min) 
                    plt.plot(step_list, variance_SN_t_min, 'o', label = 'trotterized, at t_min <= t_opt')
                    plt.plot(step_list, variance_SN_t_opt, 'o', label = 'trotterized, at t_opt')
                    if len(step_list) > 0:
                        plt.hlines(min_variance_SN_cont, min(step_list), max(step_list), linestyle='dashed', label = 'continuous, at t_opt')
                        plt.hlines(2 * min_variance_SN_cont, min(step_list), max(step_list), linestyle='dashed', label = ' x2, continuous, at t_opt', color='red')
                    plt.legend()
                    plt.ylim(bottom=0., top=1.5)
                    plt.xscale('log')
                    plt.savefig('{}/plots/variance_SN_vs_steps_N_{}_{}_{}_{}.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range))
                    plt.close()

                    fig = plt.figure()
                    plt.title(r'TFI, $J_z = -1$, $h = N \cdot J_z /2$, $N = {}$, power_law, $\alpha = {}$'.format(N, interaction_range))
                    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                    plt.xlabel(r'$t$')
                    
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h, steps)
                        if filename in os.listdir(dirname):
                            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                            variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                            t = observed_t['t']
                            plt.plot(t, variance_SN_t, label='# steps = {}'.format(steps))
                    plt.plot(2 * cont_t, cont_variance_SN_t, label='continuous')
                    plt.vlines(2 * total_T, 0., 1., linestyle='dashed', label = 'total_T', color='red')
                    plt.ylim(bottom=0., top=1.5)
                    plt.xlim(left=0., right=2 * total_T)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
                    plt.tight_layout()
                    plt.savefig('{}/plots/variance_SN_vs_t_N_{}_{}_{}_{}_all_steps.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range))
                    plt.ylim(bottom=0.8 * min(cont_variance_SN_t), top=1.2)
                    plt.yscale('log')
                    plt.tight_layout()
                    plt.savefig('{}/plots/log_variance_SN_vs_t_N_{}_{}_{}_{}_all_steps.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range))
                    plt.close()

                    print('N = {}'.format(N))

                    print('at t_opt, below 1: {}, above 1: {}'.format(step_list[variance_SN_t_opt < 1], step_list[variance_SN_t_opt > 1]))
                    
                    step_upper_vs_N_vs_version[1][N] = min(step_list[variance_SN_t_opt < 1]) if len(step_list[variance_SN_t_opt < 1]) > 0 else np.inf
                    step_lower_vs_N_vs_version[1][N] = max(step_list[variance_SN_t_opt > 1]) if len(step_list[variance_SN_t_opt > 1]) > 0 else 0.
                    print('at t_opt, below 2xmin: {}, above 2xmin: {}'.format(step_list[variance_SN_t_opt < 2 * min_variance_SN_cont],step_list[variance_SN_t_opt > 2 * min_variance_SN_cont]))
                    step_upper_vs_N_vs_version[2][N] = min(step_list[variance_SN_t_opt < 2 * min_variance_SN_cont]) if len(step_list[variance_SN_t_opt < 2 * min_variance_SN_cont]) > 0 else np.inf
                    step_lower_vs_N_vs_version[2][N] = max(step_list[variance_SN_t_opt > 2 * min_variance_SN_cont]) if len(step_list[variance_SN_t_opt > 2 * min_variance_SN_cont]) > 0 else 0.
                    print('at t_min, below 2xmin: {}, above 2xmin: {}'.format(step_list[variance_SN_t_min < 2 * min_variance_SN_cont],step_list[variance_SN_t_min > 2 * min_variance_SN_cont]))
                    step_upper_vs_N_vs_version[3][N] = min(step_list[variance_SN_t_min < 2 * min_variance_SN_cont]) if len(step_list[variance_SN_t_min < 2 * min_variance_SN_cont]) > 0 else np.inf
                    step_lower_vs_N_vs_version[3][N] = max(step_list[variance_SN_t_min > 2 * min_variance_SN_cont]) if len(step_list[variance_SN_t_min > 2 * min_variance_SN_cont]) > 0 else 0.

        total_T_vs_N_vs_range[interaction_range] = total_T_vs_N

        for version in [1,2,3]:
            step_upper_vs_N = step_upper_vs_N_vs_version[version]
            step_lower_vs_N = step_lower_vs_N_vs_version[version]
            fig = plt.figure()
            version_description = '[squeezing = 1] at t_opt' if version == 1 else '[squeezing = 2 * min squeezing] at t_opt' if version == 2 else '[squeezing = 2 * min squeezing] at t_min <= t_opt' 
            plt.title('max # steps where ' + version_description)
            plt.ylabel('# steps')
            plt.xlabel('N')
            N_list = []
            step_upper_list = []
            step_lower_list = []
            for N, step_upper in step_upper_vs_N.items():
                step_lower = step_lower_vs_N[N]
                N_list.append(N)
                step_upper_list.append(step_upper)
                step_lower_list.append(step_lower)
            plt.plot(N_list, step_upper_list, 'o', label='upper limit')
            plt.plot(N_list, step_lower_list, 'o', label='lower limit')
            plt.legend()
            plt.savefig('{}/plots/num_steps_vs_N_version_{}_{}_{}_{}.png'.format(dirname, version, interaction_shape, interaction_param_name, interaction_range))

        # for version in [1,2,3]:
        #     step_upper_vs_N = step_upper_vs_N_vs_version[version]
        #     step_lower_vs_N = step_lower_vs_N_vs_version[version]
        #     fig = plt.figure()
        #     version_description = '[squeezing = 1] at t_opt' if version == 1 else '[squeezing = 2 * min squeezing] at t_opt' if version == 2 else '[squeezing = 2 * min squeezing] at t_min <= t_opt' 
        #     plt.title('1 / (max # steps) where ' + version_description)
        #     plt.ylabel('1 / # steps')
        #     plt.xlabel('N')
        #     N_list = []
        #     step_upper_list = []
        #     step_lower_list = []
        #     for N, step_upper in step_upper_vs_N.items():
        #         step_lower = step_lower_vs_N[N]
        #         N_list.append(N)
        #         step_upper_list.append(step_upper)
        #         step_lower_list.append(step_lower)
        #     plt.plot(N_list, 1. / np.array(step_upper_list), 'o', label='upper limit')
        #     plt.plot(N_list, 1. / np.array(step_lower_list), 'o', label='lower limit')
        #     plt.plot(N_list, 1. / np.array(N_list), linestyle='dashed', label='1/N')
        #     plt.legend()
        #     plt.savefig('{}/plots/inv_num_steps_vs_N_version_{}_{}_{}_{}.png'.format(dirname, version, interaction_shape, interaction_param_name, interaction_range))
        #     plt.yscale('log')
        #     plt.xscale('log')
        #     plt.savefig('{}/plots/log_inv_num_steps_vs_N_version_{}_{}_{}_{}.png'.format(dirname, version, interaction_shape, interaction_param_name, interaction_range))

        # for version in [1,2,3]:
        #     step_upper_vs_N = step_upper_vs_N_vs_version[version]
        #     step_lower_vs_N = step_lower_vs_N_vs_version[version]
        #     fig = plt.figure()
        #     version_description = '[squeezing = 1] at t_opt' if version == 1 else '[squeezing = 2 * min squeezing] at t_opt' if version == 2 else '[squeezing = 2 * min squeezing] at t_min <= t_opt' 
        #     plt.title('Δt where ' + version_description)
        #     plt.ylabel('Δt')
        #     plt.xlabel('N')
        #     N_list = []
        #     delta_t_upper_list = []
        #     delta_t_lower_list = []
        #     for N, step_upper in step_upper_vs_N.items():
        #         step_lower = step_lower_vs_N[N]
        #         N_list.append(N)
        #         delta_t_upper_list.append(np.divide(total_T, step_lower))
        #         delta_t_lower_list.append(np.divide(total_T, step_upper))
        #     plt.plot(N_list, delta_t_upper_list, 'o', label='upper limit')
        #     plt.plot(N_list, delta_t_lower_list, 'o', label='lower limit')
        #     # plt.plot(N_list, 1. / np.power(N_list, 1.7), linestyle='dashed', label='1/ N ^ (1.7)')

        #     plt.legend()
        #     plt.savefig('{}/plots/delta_t_vs_N_version_{}_{}_{}_{}.png'.format(dirname, version, interaction_shape, interaction_param_name, interaction_range))
        #     plt.yscale('log')
        #     plt.xscale('log')
        #     plt.savefig('{}/plots/log_delta_t_vs_log_N_version_{}_{}_{}_{}.png'.format(dirname, version, interaction_shape, interaction_param_name, interaction_range))

        fig = plt.figure()
        plt.title(r'TFI, $J_z = -1$, $h = N \cdot J_z /2$, power_law, $\alpha = {}$'.format(interaction_range))
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$ at $t_{opt}$')
        plt.xlabel('# steps')
        for N in [10, 20, 50, 100, 200, 500, 1000]:
                    Jz = -1.
                    h = N * Jz / 2

                    system_size = (N, 1)
                    spin_system = sd.SpinOperators_Symmetry(system_size)
                    variance_SN_t_min = []
                    variance_SN_t_opt = []
                    step_list = []
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h, steps)
                        if filename in os.listdir(dirname):
                            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                            variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                            t = observed_t['t']
                            variance_SN_t_opt.append(variance_SN_t[-1])
                            variance_SN_t_min.append(np.min(variance_SN_t))
                            step_list.append(steps)
                    step_list = np.array(step_list)
                    variance_SN_t_opt = np.array(variance_SN_t_opt)
                    variance_SN_t_min = np.array(variance_SN_t_min) 
                    prev = plt.plot(step_list, variance_SN_t_opt, 'o', alpha=0.7, label = 'N = {}'.format(N))
                    if len(step_list) > 0:
                        plt.hlines(min_variance_SN_cont_vs_N[N], min(step_list), max(step_list), alpha=0.5, color=prev[-1].get_color(), linestyle='dashed', label = 'N = {}, continuous'.format(N))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('{}/plots/variance_SN_at_t_opt_vs_steps_{}_{}_{}_all_N.png'.format(dirname, interaction_shape, interaction_param_name, interaction_range))
        plt.close()

        fig = plt.figure()
        plt.title(r'TFI, $J_z = -1$, $h = N \cdot J_z /2$, power_law, $\alpha = {}$'.format(interaction_range))
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$ at $t_{min} <= t_{opt}$')
        plt.xlabel('# steps')
        for N in [10, 20, 50, 100, 200, 500, 1000]:
                    Jz = -1.
                    h = N * Jz / 2

                    system_size = (N, 1)
                    spin_system = sd.SpinOperators_Symmetry(system_size)
                    # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                    variance_SN_t_min = []
                    variance_SN_t_opt = []
                    # for steps in step_list:
                    step_list = []
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h, steps)
                        if filename in os.listdir(dirname):
                            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                            variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                            t = observed_t['t']
                            variance_SN_t_opt.append(variance_SN_t[-1])
                            variance_SN_t_min.append(np.min(variance_SN_t))
                            step_list.append(steps)
                    step_list = np.array(step_list)
                    variance_SN_t_opt = np.array(variance_SN_t_opt)
                    variance_SN_t_min = np.array(variance_SN_t_min) 
                    prev = plt.plot(step_list, variance_SN_t_min, 'o', alpha=0.7, label = 'N = {}'.format(N))
                    if len(step_list) > 0:
                        plt.hlines(min_variance_SN_cont_vs_N[N], min(step_list), max(step_list), alpha=0.5, color=prev[-1].get_color(), linestyle='dashed', label = 'N = {}, continuous'.format(N))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('{}/plots/variance_SN_at_t_min_vs_steps_{}_{}_{}_all_N.png'.format(dirname, interaction_shape, interaction_param_name, interaction_range))
        plt.close()

        print(total_T_vs_N_vs_range)
