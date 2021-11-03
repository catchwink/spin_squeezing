import numpy as np
import setup
import spin_dynamics as sd
import util
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    method = 'TFI'
    dirname = 'TFI_trotter_symm'
    interaction_shape = 'power_law'
    interaction_param_name = 'exp'

    total_T_vs_N = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.104, 0.0672, 0.036000000000000004, 0.0216, 0.012, 0.005600000000000001, 0.0032]))
    min_variance_SN_vs_N = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.27930329604706866-4.8871434199516e-17j), (0.16229489546823814+8.406013995130382e-17j), (0.0722320680944569-6.412969160985913e-17j), (0.03759725937879103+2.7255140213520144e-16j), (0.019384361968288702-7.563967980256765e-16j), (0.008027525988677058-2.670675983419243e-15j), (0.003950094500726353+2.233050141887525e-15j)]))

    # version 1: at t_opt < 1, version 2: at t_opt < 2xmin, version 3: at t_min < 2xmin
    step_upper_vs_N_vs_version = {1:{},2:{},3:{}}
    step_lower_vs_N_vs_version = {1:{},2:{},3:{}}

    for N in [10, 20, 50, 100, 200, 500, 1000]:
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)

        for interaction_range in [0]:

                Jz = -1.
                h = N * Jz * 0.5

                fig = plt.figure()
                plt.title(r'$N = {}$, power_law, $\alpha = {}$'.format(N, interaction_range))
                plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                plt.xlabel('# steps')
                # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                variance_SN_t_min = []
                variance_SN_t_opt = []
                # for steps in step_list:
                step_list = []
                for steps in range(10001):

                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
                    if filename in os.listdir(dirname):
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                        step_list.append(steps)
                step_list = np.array(step_list)
                variance_SN_t_opt = np.array(variance_SN_t_opt)
                variance_SN_t_min = np.array(variance_SN_t_min) 
                plt.plot(step_list, variance_SN_t_min, 'o', label = 'trotterized, at t_min <= t_opt')
                plt.plot(step_list, variance_SN_t_opt, 'o', label = 'trotterized, at t_opt')
                plt.hlines(min_variance_SN_vs_N[N], min(step_list), max(step_list), linestyle='dashed', label = 'continuous, at t_opt')
                plt.hlines(2 * min_variance_SN_vs_N[N], min(step_list), max(step_list), linestyle='dashed', label = ' x2, continuous, at t_opt', color='red')
                plt.legend()
                plt.ylim(bottom=0., top=1.5)
                plt.xscale('log')
                plt.savefig('{}/plots/variance_SN_vs_steps_N_{}_{}_{}_{}.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range))
                plt.close()
                print('N = {}'.format(N))

                print('at t_opt, below 1: {}, above 1: {}'.format(step_list[variance_SN_t_opt < 1], step_list[variance_SN_t_opt > 1]))
                
                step_upper_vs_N_vs_version[1][N] = min(step_list[variance_SN_t_opt < 1]) if len(step_list[variance_SN_t_opt < 1]) > 0 else np.inf
                step_lower_vs_N_vs_version[1][N] = max(step_list[variance_SN_t_opt > 1]) if len(step_list[variance_SN_t_opt > 1]) > 0 else 0.
                print('at t_opt, below 2xmin: {}, above 2xmin: {}'.format(step_list[variance_SN_t_opt < 2 * min_variance_SN_vs_N[N]],step_list[variance_SN_t_opt > 2 * min_variance_SN_vs_N[N]]))
                step_upper_vs_N_vs_version[2][N] = min(step_list[variance_SN_t_opt < 2 * min_variance_SN_vs_N[N]]) if len(step_list[variance_SN_t_opt < 2 * min_variance_SN_vs_N[N]]) > 0 else np.inf
                step_lower_vs_N_vs_version[2][N] = max(step_list[variance_SN_t_opt > 2 * min_variance_SN_vs_N[N]]) if len(step_list[variance_SN_t_opt > 2 * min_variance_SN_vs_N[N]]) > 0 else 0.
                print('at t_min, below 2xmin: {}, above 2xmin: {}'.format(step_list[variance_SN_t_min < 2 * min_variance_SN_vs_N[N]],step_list[variance_SN_t_min > 2 * min_variance_SN_vs_N[N]]))
                step_upper_vs_N_vs_version[3][N] = min(step_list[variance_SN_t_min < 2 * min_variance_SN_vs_N[N]]) if len(step_list[variance_SN_t_min < 2 * min_variance_SN_vs_N[N]]) > 0 else np.inf
                step_lower_vs_N_vs_version[3][N] = max(step_list[variance_SN_t_min > 2 * min_variance_SN_vs_N[N]]) if len(step_list[variance_SN_t_min > 2 * min_variance_SN_vs_N[N]]) > 0 else 0.

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
        plt.savefig('{}/plots/num_steps_vs_N_version_{}_power_law_exp_0.png'.format(dirname, version))

    for version in [1,2,3]:
        step_upper_vs_N = step_upper_vs_N_vs_version[version]
        step_lower_vs_N = step_lower_vs_N_vs_version[version]
        fig = plt.figure()
        version_description = '[squeezing = 1] at t_opt' if version == 1 else '[squeezing = 2 * min squeezing] at t_opt' if version == 2 else '[squeezing = 2 * min squeezing] at t_min <= t_opt' 
        plt.title('1 / (max # steps) where ' + version_description)
        plt.ylabel('1 / # steps')
        plt.xlabel('N')
        N_list = []
        step_upper_list = []
        step_lower_list = []
        for N, step_upper in step_upper_vs_N.items():
            step_lower = step_lower_vs_N[N]
            N_list.append(N)
            step_upper_list.append(step_upper)
            step_lower_list.append(step_lower)
        plt.plot(N_list, 1. / np.array(step_upper_list), 'o', label='upper limit')
        plt.plot(N_list, 1. / np.array(step_lower_list), 'o', label='lower limit')
        plt.plot(N_list, 1. / np.array(N_list), linestyle='dashed', label='1/N')
        plt.legend()
        plt.savefig('{}/plots/inv_num_steps_vs_N_version_{}_power_law_exp_0.png'.format(dirname, version))
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('{}/plots/log_inv_num_steps_vs_N_version_{}_power_law_exp_0.png'.format(dirname, version))

    for version in [1,2,3]:
        step_upper_vs_N = step_upper_vs_N_vs_version[version]
        step_lower_vs_N = step_lower_vs_N_vs_version[version]
        fig = plt.figure()
        version_description = '[squeezing = 1] at t_opt' if version == 1 else '[squeezing = 2 * min squeezing] at t_opt' if version == 2 else '[squeezing = 2 * min squeezing] at t_min <= t_opt' 
        plt.title('Δt where ' + version_description)
        plt.ylabel('Δt')
        plt.xlabel('N')
        N_list = []
        delta_t_upper_list = []
        delta_t_lower_list = []
        for N, step_upper in step_upper_vs_N.items():
            step_lower = step_lower_vs_N[N]
            N_list.append(N)
            delta_t_upper_list.append(np.divide(total_T_vs_N[N], step_lower))
            delta_t_lower_list.append(np.divide(total_T_vs_N[N], step_upper))
        plt.plot(N_list, delta_t_upper_list, 'o', label='upper limit')
        plt.plot(N_list, delta_t_lower_list, 'o', label='lower limit')
        plt.plot(N_list, 1. / np.power(N_list, 1.7), linestyle='dashed', label='1/ N ^ (1.7)')

        plt.legend()
        plt.savefig('{}/plots/delta_t_vs_N_version_{}_power_law_exp_0.png'.format(dirname, version))
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('{}/plots/log_delta_t_vs_log_N_version_{}_power_law_exp_0.png'.format(dirname, version))

    fig = plt.figure()
    plt.title(r'power_law, $\alpha = {}$'.format(interaction_range))
    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$ at $t_{opt}$')
    plt.xlabel('# steps')
    for N in [10, 20, 50, 100, 200, 500, 1000]:
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        for interaction_range in [0]:
                # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                variance_SN_t_min = []
                variance_SN_t_opt = []
                # for steps in step_list:
                step_list = []
                for steps in range(10001):
                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
                    if filename in os.listdir(dirname):
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                        step_list.append(steps)
                step_list = np.array(step_list)
                variance_SN_t_opt = np.array(variance_SN_t_opt)
                variance_SN_t_min = np.array(variance_SN_t_min) 
                prev = plt.plot(step_list, variance_SN_t_opt, 'o', alpha=0.7, label = 'N = {}'.format(N))
                plt.hlines(min_variance_SN_vs_N[N], min(step_list), max(step_list), alpha=0.5, color=prev[-1].get_color(), linestyle='dashed', label = 'N = {}, continuous'.format(N))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('{}/plots/variance_SN_at_t_opt_vs_steps_{}_{}_{}_all_N.png'.format(dirname, interaction_shape, interaction_param_name, interaction_range))
    plt.close()

    fig = plt.figure()
    plt.title(r'power_law, $\alpha = {}$'.format(interaction_range))
    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$ at $t_{min} <= t_{opt}$')
    plt.xlabel('# steps')
    for N in [10, 20, 50, 100, 200, 500, 1000]:
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        for interaction_range in [0]:
                # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                variance_SN_t_min = []
                variance_SN_t_opt = []
                # for steps in step_list:
                step_list = []
                for steps in range(10001):
                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
                    if filename in os.listdir(dirname):
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                        step_list.append(steps)
                step_list = np.array(step_list)
                variance_SN_t_opt = np.array(variance_SN_t_opt)
                variance_SN_t_min = np.array(variance_SN_t_min) 
                prev = plt.plot(step_list, variance_SN_t_min, 'o', alpha=0.7, label = 'N = {}'.format(N))
                plt.hlines(min_variance_SN_vs_N[N], min(step_list), max(step_list), alpha=0.5, color=prev[-1].get_color(), linestyle='dashed', label = 'N = {}, continuous'.format(N))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('{}/plots/variance_SN_at_t_min_vs_steps_{}_{}_{}_all_N.png'.format(dirname, interaction_shape, interaction_param_name, interaction_range))
    plt.close()


                