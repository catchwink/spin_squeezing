import numpy as np
import setup
import spin_dynamics as sd
import util
from collections import defaultdict
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    fig = plt.figure(figsize=(8,6))
    plt.title('power law, exp = 0')
    plt.xlabel('t')
    plt.ylabel('N * <S_a^2> / <S_x>^2')

    N_list = [10,100]
    color_idx = np.linspace(1. / len(N_list), 1., len(N_list))
    for i, N in zip(color_idx, N_list):
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        # range_list = [0,0.5,1,1.5,2,2.5,3]
        for interaction_range in [0]:
            J_list = [1.]
            for J in J_list: 
                for method in ['CT', 'CTv2']:
                    dirname = method + '_symm'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'
                    
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                    t = observed_t['t']
                    print(method, len(variance_SN_t), len(t))
                    print('N = {}, t_opt = {}'.format(N, t[np.argmin(variance_SN_t)]))
                    plt.plot(t, variance_SN_t, label=method + ', N = {}'.format(N), color=color, linestyle=linestyle)
    plt.ylim(bottom=0., top=1.)
    plt.xlim(left=0., right=1.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig('CTv2_symm/variance_SN_vs_t_power_law_exp_0_all_N.png')


    N_list = [10,100]
    for i, N in zip(color_idx, N_list):
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        fig = plt.figure()
        plt.title('N = {}, power law, exp = 0'.format(N))
        plt.xlabel('t')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        # range_list = [0,0.5,1,1.5,2,2.5,3]
        for interaction_range in [0]:
            J_list = [1.]
            for J in J_list: 
                for method in ['CT', 'CTv2']:
                    dirname = method + '_symm'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                    t = observed_t['t']
                    if method == 'CT':
                        linestyle = 'solid'
                    else:
                        linestyle = 'dashed'
                    plt.plot(t, variance_SN_t, label=method, linestyle=linestyle)
        plt.ylim(bottom=0., top=1.)
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.savefig('CTv2_symm/variance_SN_vs_t_N_{}_power_law_exp_0.png'.format(N))



