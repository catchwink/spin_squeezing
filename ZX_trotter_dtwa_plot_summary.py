import numpy as np
import setup
import spin_dynamics as sd
import util
from matplotlib import pyplot as plt
import os
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    method = 'ZX'
    dirname = 'ZX_trotter_dtwa'
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

            cont_filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format('XY', N, 'power_law', 'exp', interaction_range)
            cont_dirname = '{}_dtwa'.format('XY')
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
                    plt.title(r'$N = {}$, power_law, $\alpha = {}$'.format(N, interaction_range))
                    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                    plt.xlabel('# steps')
                    # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                    variance_SN_t_min = []
                    variance_SN_t_opt = []
                    # for steps in step_list:
                    step_list = []
                    print(N, interaction_range)
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
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
                    plt.title(r'$N = {}$, power_law, $\alpha = {}$'.format(N, interaction_range))
                    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                    plt.xlabel(r'$t$')
                    
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
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
            plt.close()

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
        plt.title(r'power_law, $\alpha = {}$'.format(interaction_range))
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$ at $t_{opt}$')
        plt.xlabel('# steps')
        for N in [10, 20, 50, 100, 200, 500, 1000]:
                    system_size = (N, 1)
                    spin_system = sd.SpinOperators_Symmetry(system_size)
                    variance_SN_t_min = []
                    variance_SN_t_opt = []
                    step_list = []
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
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
        plt.title(r'power_law, $\alpha = {}$'.format(interaction_range))
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$ at $t_{min} <= t_{opt}$')
        plt.xlabel('# steps')
        for N in [10, 20, 50, 100, 200, 500, 1000]:
                    system_size = (N, 1)
                    spin_system = sd.SpinOperators_Symmetry(system_size)
                    # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                    variance_SN_t_min = []
                    variance_SN_t_opt = []
                    # for steps in step_list:
                    step_list = []
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
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

    # total_T_vs_N_vs_range = {}
    # total_T_vs_N_vs_range[0.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.435, 0.28500000000000003, 0.159, 0.10200000000000001, 0.066, 0.03, 0.03]))
    # total_T_vs_N_vs_range[0.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.714, 0.585, 0.501, 0.429, 0.384, 0.31500000000000006, 0.28500000000000003]))
    # total_T_vs_N_vs_range[1.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.002, 1.026, 1.158, 1.2149999999999999, 1.2839999999999998, 1.5, 1.5]))
    # total_T_vs_N_vs_range[1.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2149999999999999, 1.3439999999999999, 1.5, 1.5, 1.5, 1.5, 1.5]))
    # total_T_vs_N_vs_range[2.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.251, 1.38, 1.494, 1.5, 1.5, 1.5, 1.5]))
    # total_T_vs_N_vs_range[2.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.23, 1.3499999999999999, 1.407, 1.446, 1.4609999999999999, 1.455, 1.47]))
    # total_T_vs_N_vs_range[3.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2089999999999999, 1.275, 1.341, 1.3679999999999999, 1.392, 1.395, 1.395]))
    # total_T_vs_N_vs_range[3.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2029999999999998, 1.272, 1.3079999999999998, 1.3379999999999999, 1.329, 1.335, 1.335]))
    # total_T_vs_N_vs_range[4.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2, 1.218, 1.293, 1.2959999999999998, 1.293, 1.29, 1.305]))
    # total_T_vs_N_vs_range[4.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2149999999999999, 1.2209999999999999, 1.269, 1.299, 1.293, 1.29, 1.29]))
    # total_T_vs_N_vs_range[5.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.146, 1.212, 1.2389999999999999, 1.26, 1.2839999999999998, 1.275, 1.275]))
    # total_T_vs_N_vs_range[5.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.1669999999999998, 1.2209999999999999, 1.242, 1.257, 1.272, 1.26, 1.26]))
    # total_T_vs_N_vs_range[6.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.1789999999999998, 1.2209999999999999, 1.242, 1.254, 1.2839999999999998, 1.275, 1.275]))

    # min_variance_SN_vs_N_vs_range = {}
    # min_variance_SN_vs_N_vs_range[0.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.3187557522594195+0j), (0.200022218418662+0j), (0.10387926548759195+0j), (0.061005970485207894+0j), (0.03694406317866285+0j), (0.021780251594424052+0j), (0.015858198602035235+0j)]))
    # min_variance_SN_vs_N_vs_range[0.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.32012726265203534+0j), (0.2030538881240366+0j), (0.10094572494225816+0j), (0.06221334798618659+0j), (0.0364497050343903+0j), (0.01936909544639078+0j), (0.011771853564905451+0j)]))
    # min_variance_SN_vs_N_vs_range[1.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.357197774179694+0j), (0.2241117260402238+0j), (0.10626136720553626+0j), (0.0635088101946351+0j), (0.04184846902552937+0j), (0.02102479698899758+0j), (0.013495801267634496+0j)]))
    # min_variance_SN_vs_N_vs_range[1.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.3933944660425496+0j), (0.2900270280232179+0j), (0.17937017653234508+0j), (0.13530227426892408+0j), (0.10982031905631748+0j), (0.08598025884709004+0j), (0.07966650086971928+0j)]))
    # min_variance_SN_vs_N_vs_range[2.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.4686631231913616+0j), (0.37997612366298505+0j), (0.31148836017962556+0j), (0.29074749834115077+0j), (0.26510525835127846+0j), (0.25680746034360835+0j), (0.25076795905376076+0j)]))
    # min_variance_SN_vs_N_vs_range[2.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5112363951692522+0j), (0.45885330972233723+0j), (0.40556667824049386+0j), (0.39569616698400506+0j), (0.3941256769047173+0j), (0.3762092429390486+0j), (0.3794000900364605+0j)]))
    # min_variance_SN_vs_N_vs_range[3.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5557692684073696+0j), (0.5052593980741875+0j), (0.4699810827123452+0j), (0.4635518388236521+0j), (0.45952146580324327+0j), (0.4664608019792977+0j), (0.45285551274802716+0j)]))
    # min_variance_SN_vs_N_vs_range[3.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5670255492221451+0j), (0.5278939554603594+0j), (0.5097251298257347+0j), (0.5007206937682704+0j), (0.49589922625064564+0j), (0.5000080882782595+0j), (0.49524671809143817+0j)]))
    # min_variance_SN_vs_N_vs_range[4.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5833091311772926+0j), (0.5455515351479246+0j), (0.5422806540405265+0j), (0.516929782738086+0j), (0.5219163884523763+0j), (0.5245692207638514+0j), (0.506836652495189+0j)]))
    # min_variance_SN_vs_N_vs_range[4.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5802179929807516+0j), (0.5642351535321916+0j), (0.5549683350644242+0j), (0.5478956273715833+0j), (0.5273184938235134+0j), (0.5298573107897742+0j), (0.5411648684944769+0j)]))
    # min_variance_SN_vs_N_vs_range[5.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6078529093690566+0j), (0.5782169190664381+0j), (0.5486437548300399+0j), (0.5507535087926309+0j), (0.5317498472660189+0j), (0.5465503075020792+0j), (0.5420547699510118+0j)]))
    # min_variance_SN_vs_N_vs_range[5.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6161220525169849+0j), (0.5840808588220184+0j), (0.5594419743283089+0j), (0.5555948987375099+0j), (0.5548465331215789+0j), (0.5406403473746832+0j), (0.5676516028848497+0j)]))
    # min_variance_SN_vs_N_vs_range[6.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6190377723445925+0j), (0.5975470912668818+0j), (0.5537568966317522+0j), (0.5702852666361092+0j), (0.5649508158185218+0j), (0.5497115711221022+0j), (0.5557429433142047+0j)]))
