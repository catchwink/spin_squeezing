import numpy as np
import setup
import spin_dynamics as sd
import util
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    dirname = 'XY_dtwa'

    variance_SN_vs_range_vs_N = defaultdict(dict)
    t_vs_range_vs_N = defaultdict(dict)
    variance_SN_vs_N_vs_range = defaultdict(dict)
    t_vs_N_vs_range = defaultdict(dict)
    method = 'XY'
    for N in [10,20,50,100,200,500,1000]:
        fig = plt.figure()
        title = title = r'XY, power law, $N = {}$'.format(N)
        plt.title(title)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
        plt.ylim(bottom=0., top=1.)
        range_list = [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4,4.5,5.,5.5,6.]
        color_idx = np.linspace(1. / len(range_list), 1., len(range_list))
        for i, interaction_range in zip(color_idx, range_list):
            J_list = [1.]
            for J in J_list: 
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, 'power_law', 'exp', interaction_range)
                if filename in os.listdir(dirname):
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']

                    variance_SN_vs_range_vs_N[N][interaction_range] = variance_SN_t
                    t_vs_range_vs_N[N][interaction_range] = t
                    variance_SN_vs_N_vs_range[interaction_range][N] = variance_SN_t
                    t_vs_N_vs_range[interaction_range][N] = t
                    plt.plot(t, variance_SN_t, label = r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        plt.legend(fontsize='x-small')
        plt.tight_layout()
        plt.savefig('XY_dtwa/plots/variance_SN_vs_t_N_{}_all_ranges.png'.format(N))
        plt.close()

    # fig = plt.figure()
    # title = 'XY, power law'
    # plt.title(title)
    # plt.xlabel(r'$\alpha$')
    # plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
    # plt.ylim(bottom=0., top=1.)
    # color_idx = np.linspace(1. / len(variance_SN_vs_range_vs_N), 1., len(variance_SN_vs_range_vs_N))
    # for i, (N, variance_SN_vs_range) in zip(color_idx, variance_SN_vs_range_vs_N.items()):
    #     range_list = []
    #     min_variance_SN_list = []
    #     for interaction_range, variance_SN_t in variance_SN_vs_range.items():
    #         range_list.append(interaction_range)
    #         min_variance_SN_list.append(min(variance_SN_t))
    #     plt.plot(range_list, min_variance_SN_list, marker='o', label=r'$N = {}$'.format(N), color=plt.cm.get_cmap('Reds')(i))
    # plt.legend(fontsize='x-small')
    # plt.savefig('XY_dtwa/plots/min_variance_SN_vs_range_all_N.png'.format(N))
    # plt.yscale('log')
    # plt.ylim(bottom=None, top=None)
    # plt.tight_layout()
    # plt.savefig('XY_dtwa/plots/log_min_variance_SN_vs_range_all_N.png'.format(N))
    # plt.close()


    fig = plt.figure()
    title = 'XY, power law'
    plt.title(title)
    plt.xlabel(r'$N$')
    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
    plt.ylim(bottom=0., top=1.)
    color_idx = np.linspace(1. / len(variance_SN_vs_N_vs_range), 1., len(variance_SN_vs_N_vs_range))
    b_vs_range = {}
    for i, (interaction_range, variance_SN_vs_N) in zip(color_idx, variance_SN_vs_N_vs_range.items()):
        N_list = []
        min_variance_SN_list = []
        for N, variance_SN_t in variance_SN_vs_N.items():
            N_list.append(N)
            min_variance_SN_list.append(min(variance_SN_t))
        prev = plt.plot(N_list, min_variance_SN_list, marker='o', label=r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        def fn(N, a, b):
            return a * np.power(N, b)
        popt, pcov = curve_fit(fn, np.array(N_list), min_variance_SN_list)
        b_vs_range[interaction_range] = tuple(popt)[1]
        plt.plot(N_list, fn(N_list, *popt), label=r'$%5.3f \cdot N^{%5.3f}$' % tuple(popt), linestyle='dashed', color=prev[-1].get_color())
    plt.legend(fontsize='x-small', ncol=2)
    plt.savefig('XY_dtwa/plots/min_variance_SN_vs_N_all_ranges.png'.format(N))
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(bottom=None, top=None)
    plt.tight_layout()
    plt.savefig('XY_dtwa/plots/log_min_variance_SN_vs_log_N_all_ranges.png'.format(N))
    plt.close()

    def fn2(N, a, b, c, d):
        return a / (b + np.power(N, c)) + d
    fig = plt.figure()
    plt.title(r'XY, power law, $\min( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 ) = A \cdot N ^ \beta$')
    plt.ylabel(r'$-\beta$')
    plt.xlabel(r'$\alpha$')
    range_list, b_list = [], []
    for interaction_range, b in b_vs_range.items():
        range_list.append(interaction_range)
        b_list.append(-b)
    plt.plot(range_list, b_list, 'o')
    popt0, pcov0 = curve_fit(fn2, np.array(range_list), b_list)
    plt.plot(np.logspace(np.log(0.5), np.log(3), num=1000), fn2(np.logspace(np.log(0.5), np.log(3), num=1000), *popt0), linestyle='dashed', label=r'$%5.3f / (%5.3f + \alpha^{%5.3f}) + %5.3f$' % tuple(popt0))
    plt.legend(fontsize='x-small')
    plt.ylim(0.001, 1)
    plt.tight_layout()
    plt.savefig('{}/plots/b_vs_range_for_min_variance_SN.png'.format('XY_dtwa'))
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('{}/plots/log_b_vs_log_range_for_min_variance_SN.png'.format('XY_dtwa'))
    plt.close()

    # fig = plt.figure()
    # title = 'XY, power law'
    # plt.title(title)
    # plt.xlabel(r'$\alpha$')
    # plt.ylabel(r'$t_{opt}$')
    # color_idx = np.linspace(1. / len(t_vs_range_vs_N), 1., len(t_vs_range_vs_N))
    # for i, (N, t_vs_range) in zip(color_idx, t_vs_range_vs_N.items()):
    #     range_list = []
    #     t_opt_list = []
    #     for interaction_range, t in t_vs_range.items():
    #         range_list.append(interaction_range)
    #         t_opt_list.append(t[np.argmin(variance_SN_vs_range_vs_N[N][interaction_range])])
    #     plt.plot(range_list, t_opt_list, marker='o', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
    # plt.legend(fontsize='x-small')
    # plt.tight_layout()
    # plt.savefig('XY_dtwa/plots/t_opt_vs_range_all_N.png'.format(N))
    # plt.close()

    fig = plt.figure()
    title = 'XY, power law'
    plt.title(title)
    plt.xlabel(r'$N$')
    plt.ylabel(r'$t_{opt}$')
    color_idx = np.linspace(1. / len(t_vs_N_vs_range), 1., len(t_vs_N_vs_range))
    m_vs_range = {}
    for i, (interaction_range, t_vs_N) in zip(color_idx, t_vs_N_vs_range.items()):
        N_list = []
        t_opt_list = []
        for N, t in t_vs_N.items():
            N_list.append(N)
            t_opt_list.append(t[np.argmin(variance_SN_vs_N_vs_range[interaction_range][N])])
        prev = plt.plot(N_list, t_opt_list, marker='o', label=r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        def fn(N, a, b):
            return a * np.power(N, b)
        popt, pcov = curve_fit(fn, np.array(N_list), t_opt_list)
        m_vs_range[interaction_range] = tuple(popt)[1]
        plt.plot(N_list, fn(N_list, *popt), label=r'$%5.3f \cdot N^{%5.3f}$' % tuple(popt), linestyle='dashed', color=prev[-1].get_color())
    plt.plot(N_list, 1 / np.power(N_list, 2/3), label=r'$1 / N^{2/3}$', linestyle='dashed')
    plt.legend(fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.savefig('XY_dtwa/plots/t_opt_vs_N_all_ranges.png'.format(N))
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('XY_dtwa/plots/log_t_opt_vs_log_N_all_ranges.png'.format(N))
    plt.close()


    fig = plt.figure()
    plt.title(r'XY, power law, $t_{opt} = A \cdot N ^ \mu$')
    plt.ylabel(r'$-\mu$')
    plt.xlabel(r'$\alpha$')
    range_list, m_list = [], []
    for interaction_range, m in m_vs_range.items():
        range_list.append(interaction_range)
        m_list.append(-b)
    plt.plot(range_list, b_list, 'o')
    plt.legend(fontsize='x-small')
    plt.tight_layout()
    plt.savefig('{}/plots/m_vs_range_for_t_opt.png'.format('XY_dtwa'))
    plt.close()

