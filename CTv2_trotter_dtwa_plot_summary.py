import numpy as np
import setup
import spin_dynamics as sd
import util
from matplotlib import pyplot as plt
import os
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    method = 'CTv2'
    dirname = 'CTv2_trotter_dtwa'
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

            cont_filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, 1.)
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
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, 1., steps)
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
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, 1., steps)
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
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, 1., steps)
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
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, 1., steps)
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
    # total_T_vs_N_vs_range[0.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.234, 0.14700000000000002, 0.07500000000000001, 0.045000000000000005, 0.027, 0.015]))
    # total_T_vs_N_vs_range[0.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.366, 0.3, 0.222, 0.17700000000000002, 0.138, 0.09000000000000001]))
    # total_T_vs_N_vs_range[1.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.47400000000000003, 0.468, 0.441, 0.423, 0.399, 0.37500000000000006]))
    # total_T_vs_N_vs_range[1.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.522, 0.546, 0.5670000000000001, 0.5790000000000001, 0.585, 0.5850000000000001]))
    # total_T_vs_N_vs_range[2.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.531, 0.552, 0.5730000000000001, 0.588, 0.591, 0.6000000000000001]))
    # total_T_vs_N_vs_range[2.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.528, 0.543, 0.558, 0.5670000000000001, 0.5670000000000001, 0.5700000000000001]))
    # total_T_vs_N_vs_range[3.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.522, 0.534, 0.546, 0.555, 0.552, 0.555]))
    # total_T_vs_N_vs_range[3.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.516, 0.528, 0.54, 0.546, 0.543, 0.54]))
    # total_T_vs_N_vs_range[4.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.513, 0.522, 0.534, 0.54, 0.537, 0.54]))
    # total_T_vs_N_vs_range[4.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.51, 0.522, 0.531, 0.534, 0.531, 0.54]))
    # total_T_vs_N_vs_range[5.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.507, 0.519, 0.528, 0.531, 0.528, 0.54]))
    # total_T_vs_N_vs_range[5.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.507, 0.516, 0.528, 0.531, 0.528, 0.54]))
    # total_T_vs_N_vs_range[6.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.504, 0.516, 0.525, 0.531, 0.525, 0.525]))

    # min_variance_SN_vs_N_vs_range = {}
    # min_variance_SN_vs_N_vs_range[0.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.2643255580751745+0j), (0.15313228644981478+0j), (0.06645637577770626+0j), (0.0324331560189242+0j), (0.017792362961227135+0j), (0.02502876242185012+0j)]))
    # min_variance_SN_vs_N_vs_range[0.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.28650358797666853+0j), (0.1719298104417085+0j), (0.07971308127327577+0j), (0.04189082641076615+0j), (0.02559479572609471+0j), (0.0143154602956668+0j)]))
    # min_variance_SN_vs_N_vs_range[1.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.34236548314903054+0j), (0.23243575338764622+0j), (0.12952100824626334+0j), (0.08189521483925215+0j), (0.05819900749241233+0j), (0.03596111997718744+0j)]))
    # min_variance_SN_vs_N_vs_range[1.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.41406874237788727+0j), (0.33160761840644665+0j), (0.24199333368780807+0j), (0.1994494970849258+0j), (0.18146255073528225+0j), (0.157888485919671+0j)]))
    # min_variance_SN_vs_N_vs_range[2.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.47737681283347466+0j), (0.42499303825583495+0j), (0.36085990140380625+0j), (0.3338841803006385+0j), (0.33425625652176305+0j), (0.3184372932364217+0j)]))
    # min_variance_SN_vs_N_vs_range[2.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5233441376359026+0j), (0.4895663517594555+0j), (0.4392642397071502+0j), (0.42026502956574924+0j), (0.4289846991973973+0j), (0.41534410807409555+0j)]))
    # min_variance_SN_vs_N_vs_range[3.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5543456629270804+0j), (0.5305419089906293+0j), (0.4860703140423861+0j), (0.47035070236237325+0j), (0.48218789178747984+0j), (0.4688024991395717+0j)]))
    # min_variance_SN_vs_N_vs_range[3.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5748434062412726+0j), (0.5563941308815435+0j), (0.5144597233354304+0j), (0.5002905699965098+0j), (0.5135441365460016+0j), (0.5001821163379299+0j)]))
    # min_variance_SN_vs_N_vs_range[4.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5884076901222134+0j), (0.5729562493205971+0j), (0.5322278168220924+0j), (0.5189160184758863+0j), (0.5329269392524856+0j), (0.5193831301201627+0j)]))
    # min_variance_SN_vs_N_vs_range[4.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5974447038777907+0j), (0.583753218504027+0j), (0.5436659459059768+0j), (0.5308689932680419+0j), (0.5453262969078542+0j), (0.5317034047487654+0j)]))
    # min_variance_SN_vs_N_vs_range[5.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6035196046707608+0j), (0.5909026938913102+0j), (0.5511837128940503+0j), (0.5387179341886948+0j), (0.5534559821659437+0j), (0.5397925143973256+0j)]))
    # min_variance_SN_vs_N_vs_range[5.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6076387182803387+0j), (0.5957042870831835+0j), (0.5562083306019326+0j), (0.5439515365226747+0j), (0.5588753874783245+0j), (0.5451916162385702+0j)]))
    # min_variance_SN_vs_N_vs_range[6.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6104514516147783+0j), (0.5989561122723824+0j), (0.5595995956397998+0j), (0.5474924149519805+0j), (0.562537552004784+0j), (0.5488174006297087+0j)]))


    #                 