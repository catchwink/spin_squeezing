import numpy as np
import setup
import spin_dynamics as sd
import util
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    dirname = 'inhomogeneous_trotter_dtwa'
    for method in ['CTv2', 'ZX']:
    # for method in ['CTv2']:

        # version 1: at t_opt < 1, version 2: at t_opt < 2xmin, version 3: at t_min < 2xmin
        step_upper_vs_N_vs_version = {1:{},2:{},3:{}}
        step_lower_vs_N_vs_version = {1:{},2:{},3:{}}
        
        min_variance_SN_cont_vs_N = {}
        total_T_vs_N = {}
        for N in [1000]:
            

            cont_method = 'XY' if method == 'ZX' else method
            cont_filename = 'observables_vs_t_{}_inhomogeneous_N_{}_RD_J_{}'.format(cont_method, N, 1.)
            cont_dirname = 'inhomogeneous_dtwa'
            if cont_filename in os.listdir(cont_dirname):
                cont_observed_t = util.read_observed_t('{}/{}'.format(cont_dirname, cont_filename))
                cont_variance_SN_t, cont_variance_norm_t, cont_angle_t, cont_t = cont_observed_t['min_variance_SN'], cont_observed_t['min_variance_norm'], cont_observed_t['opt_angle'], cont_observed_t['t']
                
                if len(argrelextrema(cont_variance_SN_t, np.less)[0]) > 0:
                    idx_first_cont_variance_SN = argrelextrema(cont_variance_SN_t, np.less)[0][0]
                    total_T = cont_t[idx_first_cont_variance_SN]
                    total_T_vs_N[N] = total_T
                    min_variance_SN_cont = cont_variance_SN_t[idx_first_cont_variance_SN]
                    min_variance_SN_cont_vs_N[N] = min_variance_SN_cont

                    fig = plt.figure()
                    plt.title(r'Inhomogeneous (Gaussian) distribution, Rydberg dressing, {}, $N = {}$'.format(method, N))
                    plt.xlabel('# steps')
                    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                        
                    variance_SN_t_min = []
                    variance_SN_t_opt = []

                    step_list = []
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                            filename = 'observables_vs_t_{}_inhomogeneous_N_{}_RD_J_{}_steps_{}'.format(method, N, 1., steps)
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
                    plt.hlines(min_variance_SN_cont, min(step_list), max(step_list), linestyle='dashed', label = 'continuous, at t_opt')
                    plt.hlines(2 * min_variance_SN_cont, min(step_list), max(step_list), linestyle='dashed', label = ' x2, continuous, at t_opt', color='red')
                    plt.legend()
                    plt.ylim(bottom=0., top=1.5)
                    plt.savefig('{}/plots/{}_variance_SN_vs_steps_N_{}_RD_J_{}.png'.format(dirname, method, N, 1.))
                    plt.xscale('log')
                    plt.savefig('{}/plots/{}_variance_SN_vs_log_steps_N_{}_RD_J_{}.png'.format(dirname, method, N, 1.))
                    plt.close()


                    fig = plt.figure()
                    plt.title(r'Inhomogeneous (Gaussian) distribution, Rydberg dressing, {}, $N = {}$'.format(method, N))
                    plt.xlabel('t')
                    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                    
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                            filename = 'observables_vs_t_{}_inhomogeneous_N_{}_RD_J_{}_steps_{}'.format(method, N, 1., steps)
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
                    plt.savefig('{}/plots/{}_variance_SN_vs_t_N_{}_RD_J_{}_all_steps.png'.format(dirname, method, N, 1.))
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

        fig = plt.figure()
        plt.title(r'{}, power law'.format(method))
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$ at $t_{opt}$')
        plt.xlabel('# steps')
        for N in [1000]:
                    system_size = (N, 1)
                    spin_system = sd.SpinOperators_Symmetry(system_size)
                    variance_SN_t_min = []
                    variance_SN_t_opt = []
                    step_list = []
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_inhomogeneous_N_{}_RD_J_{}_steps_{}'.format(method, N, 1., steps)
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
        plt.savefig('{}/plots/{}_variance_SN_at_t_opt_vs_steps_RD_J_{}_all_N.png'.format(dirname, method, 1.))
        plt.close()

        fig = plt.figure()
        plt.title(r'{}, power law'.format(method))
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$ at $t_{min} <= t_{opt}$')
        plt.xlabel('# steps')
        for N in [1000]:
                    system_size = (N, 1)
                    spin_system = sd.SpinOperators_Symmetry(system_size)
                    # step_list = [1, 5, 10, 100, 1000, 5000, 10000]
                    variance_SN_t_min = []
                    variance_SN_t_opt = []
                    # for steps in step_list:
                    step_list = []
                    for steps in list(range(16)) + [20, 50, 100, 200, 500, 1000]:
                        filename = 'observables_vs_t_{}_inhomogeneous_N_{}_RD_J_{}_steps_{}'.format(method, N, 1., steps)
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
        plt.savefig('{}/plots/{}_variance_SN_at_t_min_vs_steps_RD_J_{}_all_N.png'.format(dirname, method, 1.))
        plt.close()
