import numpy as np
import setup
import spin_dynamics as sd
import util
import os
from matplotlib import pyplot as plt

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    method = 'XXZ'
    dirname = 'XXZ_trotter_symm'
    interaction_shape = 'power_law'
    interaction_param_name = 'exp'


    min_variance_SN_vs_Jeff_vs_N = {}
    min_variance_SN_vs_Jeff_vs_N[10] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [(0.3098620600201123-6.210332529173473e-18j), (0.3098620600201132-1.3389305523251998e-17j), (0.3098620600201125-1.133247767151895e-16j), (0.309862060020112-4.7163077004882915e-17j), (0.3098620600201111+1.0284139258665255e-17j), (0.30986206002010946-3.3473263808129965e-18j), (0.309862060020109+5.02098957121948e-17j), (0.3098620600201057-4.422026567607081e-17j), (0.3098620600201161+1.2904985290799483e-17j), (0.3098620600201374+1.3389305523252154e-17j), (0.3098620600201219+4.637824909892992e-17j), (0.3098620600201102+1.3389305523251969e-17j), (0.3098620600201122+4.843202324525398e-19j)]))
    min_variance_SN_vs_Jeff_vs_N[20] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [(0.197904608128053+6.729414593135079e-18j), (0.1979046081280505+4.4671994118392815e-17j), (0.1979046081280391+9.542588928455285e-17j), (0.197904608128051+3.0585886914273393e-17j), (0.1979046081280696-1.3418358131229772e-17j), (0.19790460812807933+7.523764550514045e-17j), (0.197904608128045+2.5353108184524165e-17j), (0.19790460812820937+7.376928138525349e-18j), (0.19790460812842545-4.2817644320081147e-17j), (0.19790460812841915+7.894634510199715e-17j), (0.19790460812820815+1.7167528783031438e-17j), (0.19790460812804953+2.2335997059197354e-17j), (0.19790460812807825-5.0808108348777734e-17j)]))
    min_variance_SN_vs_Jeff_vs_N[50] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [(0.10404104767211551-6.434594715045042e-17j), (0.1040410476721126+3.166740446923391e-17j), (0.10404104767216804+1.106150192297195e-17j), (0.10404104767207098-2.5005788772546442e-17j), (0.10404104767225014+1.7430358561957948e-17j), (0.1040410476720351-8.568799752140854e-18j), (0.10404104767221255-2.977798908421741e-17j), (0.10404104767132104-3.077135862605526e-17j), (0.10404104767450681+1.3203701861075141e-17j), (0.10404104767370886-2.9208114548860844e-17j), (0.10404104767159146+6.130789836009147e-17j), (0.10404104767212181+6.0960848336716116e-18j), (0.10404104767200267-5.3049626335761026e-17j)]))
    min_variance_SN_vs_Jeff_vs_N[100] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [(0.06294637587196789+0j), (0.06294637587189753+2.6817228358417172e-17j), (0.06294637587197366-4.989875780120263e-17j), (0.062946375872271-5.731792759782655e-17j), (0.06294637587194826+8.482843067236749e-17j), (0.06294637587256796+2.1744900845414133e-17j), (0.06294637587474747+3.5262150590491015e-17j), (0.06294637587332222+1.7589181990438203e-17j), (0.06294637586899-2.1594371974499397e-17j), (0.06294637586339434+3.54586979921477e-17j), (0.06294637587038379-2.1058998752769067e-17j), (0.06294637587126652-1.498627597255858e-17j), (0.0629463758717136-3.5262150591414185e-17j)]))
    min_variance_SN_vs_Jeff_vs_N[200] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [(0.038034604051520156-6.993764463054847e-17j), (0.03803460405151716-1.0450565542987519e-17j), (0.038034604051523146-1.9570940302615373e-17j), (0.03803460405152942-2.352526696050127e-17j), (0.038034604051502136-3.200010721411894e-17j), (0.038034604051516395-1.0938739095832785e-17j), (0.03803460405151751+1.5915578871770942e-17j), (0.038034604051508734+1.3093522233178644e-17j), (0.03803460405134211-5.326327234503728e-18j), (0.03803460405139728+2.992444148720986e-18j), (0.03803460405150056+7.679031135892815e-18j), (0.0380346040515049+4.228172328653707e-17j), (0.038034604051513494+6.729033172123366e-17j)]))

    total_T_vs_J_eff_vs_N = {}
    total_T_vs_J_eff_vs_N[10] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [2.0, 2.5, 3.3333333333333335, 5.0, 10.0, 14.4, 14.399999999999999, 12.8, 12.8, 12.8, 12.8, 14.399999999999999, 14.4]))
    total_T_vs_J_eff_vs_N[20] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [1.36, 1.7, 2.2666666666666666, 3.4, 6.8, 13.6, 14.399999999999999, 12.8, 12.8, 12.8, 12.8, 14.399999999999999, 13.6]))
    total_T_vs_J_eff_vs_N[50] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [0.7999999999999999, 0.9999999999999999, 1.3333333333333333, 1.9999999999999998, 3.9999999999999996, 7.999999999999999, 14.399999999999999, 12.8, 12.8, 12.8, 12.8, 14.399999999999999, 7.999999999999999]))
    total_T_vs_J_eff_vs_N[100] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [0.48000000000000004, 0.6, 0.7999999999999999, 1.2, 2.4, 4.8, 9.6, 12.8, 12.8, 12.8, 12.8, 9.6, 4.8]))
    total_T_vs_J_eff_vs_N[200] = dict(zip([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005, 0.01], [0.32, 0.4, 0.5333333333333333, 0.8, 1.6, 3.2, 6.4, 12.8, 12.8, 12.8, 12.8, 6.4, 3.2]))

    # version 1: at t_opt < 1, version 2: at t_opt < 2xmin, version 3: at t_min < 2xmin
    step_upper_vs_N_vs_version = {1:{},2:{},3:{}}
    step_lower_vs_N_vs_version = {1:{},2:{},3:{}}

    for N in [10, 20, 50, 100, 200]:
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)

        for interaction_range in [0]:
            J_eff_list = [-0.1]
            for J_eff in J_eff_list:
                fig = plt.figure()
                plt.title('N = {}, power_law, exp = {}, J_eff = {}'.format(N, interaction_range, J_eff))
                plt.ylabel('N * <S_a^2> / <S_x>^2')
                plt.xlabel('# steps')
                # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                step_list = [1, 5, 10, 100, 1000]
                variance_SN_t_min = []
                variance_SN_t_opt = []
                step_list = []
                for steps in range(10001):
                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_J_eff_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff, steps)
                    if filename in os.listdir(dirname):
                        step_list.append(steps)
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
                    if filename in os.listdir(dirname):
                        step_list.append(steps)
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                step_list = np.array(step_list)
                variance_SN_t_opt = np.array(variance_SN_t_opt)
                variance_SN_t_min = np.array(variance_SN_t_min)
                plt.plot(step_list, variance_SN_t_min, 'o', label = 'trotterized, at t_min <= t_opt')
                plt.plot(step_list, variance_SN_t_opt, 'o', label = 'trotterized, at t_opt')
                plt.hlines(min_variance_SN_vs_Jeff_vs_N[N][J_eff], min(step_list), max(step_list), linestyle='dashed', label = 'continuous, at t_opt')
                plt.hlines(2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff], min(step_list), max(step_list), linestyle='dashed', label = ' x2, continuous, at t_opt', color='red')
                plt.legend()
                plt.ylim(bottom=0., top=1.5)
                plt.xscale('log')
                plt.savefig('{}/plots/variance_SN_vs_steps_N_{}_{}_{}_{}_J_eff_{}.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range, J_eff))
                plt.close()
                print('N = {}, J_eff = {}'.format(N, J_eff))

                print('at t_opt, below 1: {}, above 1: {}'.format(step_list[variance_SN_t_opt < 1], step_list[variance_SN_t_opt > 1]))
                step_upper_vs_N_vs_version[1][N] = min(step_list[variance_SN_t_opt < 1])
                step_lower_vs_N_vs_version[1][N] = max(step_list[variance_SN_t_opt > 1])
                print('at t_opt, below 2xmin: {}, above 2xmin: {}'.format(step_list[variance_SN_t_opt < 2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff]],step_list[variance_SN_t_opt > 2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff]]))
                step_upper_vs_N_vs_version[2][N] = min(step_list[variance_SN_t_opt < 2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff]])
                step_lower_vs_N_vs_version[2][N] = max(step_list[variance_SN_t_opt > 2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff]])
                print('at t_min, below 2xmin: {}, above 2xmin: {}'.format(step_list[variance_SN_t_min < 2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff]],step_list[variance_SN_t_min > 2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff]]))
                step_upper_vs_N_vs_version[3][N] = min(step_list[variance_SN_t_min < 2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff]])
                step_lower_vs_N_vs_version[3][N] = max(step_list[variance_SN_t_min > 2 * min_variance_SN_vs_Jeff_vs_N[N][J_eff]])

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
        plt.savefig('{}/plots/num_steps_vs_N_version_{}_power_law_exp_0_J_eff_-0.1.png'.format(dirname, version))

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
        plt.savefig('{}/plots/inv_num_steps_vs_N_version_{}_power_law_exp_0_J_eff_-0.1.png'.format(dirname, version))
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('{}/plots/log_inv_num_steps_vs_N_version_{}_power_law_exp_0_J_eff_-0.1.png'.format(dirname, version))

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
            delta_t_upper_list.append(total_T_vs_J_eff_vs_N[N][-0.1] / step_lower)
            delta_t_lower_list.append(total_T_vs_J_eff_vs_N[N][-0.1] / step_upper)
        plt.plot(N_list, delta_t_upper_list, 'o', label='upper limit')
        plt.plot(N_list, delta_t_lower_list, 'o', label='lower limit')
        plt.plot(N_list, 1. / np.power(N_list, 1.7), linestyle='dashed', label='1/ N ^ (1.7)')

        plt.legend()
        plt.savefig('{}/plots/delta_t_vs_N_version_{}_power_law_exp_0_J_eff_-0.1.png'.format(dirname, version))
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('{}/plots/log_delta_t_vs_log_N_version_{}_power_law_exp_0_J_eff_-0.1.png'.format(dirname, version))

    fig = plt.figure()
    plt.title('power_law, exp = {}'.format(interaction_range))
    plt.ylabel('N * <S_a^2> / <S_x>^2 at t_opt')
    plt.xlabel('# steps')
    for N in [10, 20, 50, 100, 200]:
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        for interaction_range in [0]:
                # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                variance_SN_t_min = []
                variance_SN_t_opt = []
                # for steps in step_list:
                step_list = []
                for steps in range(10001):
                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_J_eff_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff, steps)
                    if filename in os.listdir(dirname):
                        step_list.append(steps)
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
                    if filename in os.listdir(dirname):
                        step_list.append(steps)
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                step_list = np.array(step_list)
                variance_SN_t_opt = np.array(variance_SN_t_opt)
                variance_SN_t_min = np.array(variance_SN_t_min) 
                prev = plt.plot(step_list, variance_SN_t_opt, 'o', label = 'N = {}'.format(N))
                plt.hlines(min_variance_SN_vs_Jeff_vs_N[N][-0.1], min(step_list), max(step_list), color=prev[-1].get_color(), linestyle='dashed', label = 'N = {}, continuous'.format(N))
    plt.legend()
    plt.ylim(bottom=0., top=1.5)
    plt.xscale('log')
    plt.savefig('{}/plots/variance_SN_at_t_opt_vs_steps_{}_{}_{}_J_eff_-0.1_all_N.png'.format(dirname, interaction_shape, interaction_param_name, interaction_range))
    plt.close()

    fig = plt.figure()
    plt.title('power_law, exp = {}'.format(interaction_range))
    plt.ylabel('N * <S_a^2> / <S_x>^2 at t_min <= t_opt')
    plt.xlabel('# steps')
    for N in [10, 20, 50, 100, 200]:
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        for interaction_range in [0]:
                # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                variance_SN_t_min = []
                variance_SN_t_opt = []
                # for steps in step_list:
                step_list = []
                for steps in range(10001):
                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_J_eff_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J_eff, steps)
                    if filename in os.listdir(dirname):
                        step_list.append(steps)
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                    filename = 'observables_vs_t_trotter_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps)
                    if filename in os.listdir(dirname):
                        step_list.append(steps)
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']
                        variance_SN_t_opt.append(variance_SN_t[-1])
                        variance_SN_t_min.append(np.min(variance_SN_t))
                step_list = np.array(step_list)
                variance_SN_t_opt = np.array(variance_SN_t_opt)
                variance_SN_t_min = np.array(variance_SN_t_min) 
                prev = plt.plot(step_list, variance_SN_t_min, 'o', label = 'N = {}'.format(N))
                plt.hlines(min_variance_SN_vs_Jeff_vs_N[N][-0.1], min(step_list), max(step_list), color=prev[-1].get_color(), linestyle='dashed', label = 'N = {}, continuous'.format(N))
    plt.legend()
    plt.ylim(bottom=0., top=1.5)
    plt.xscale('log')
    plt.savefig('{}/plots/variance_SN_at_t_min_vs_steps_{}_{}_{}_J_eff_-0.1_all_N.png'.format(dirname, interaction_shape, interaction_param_name, interaction_range))
    plt.close()
