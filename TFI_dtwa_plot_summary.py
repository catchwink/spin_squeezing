import numpy as np
import setup
import spin_dynamics as sd
import util
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    dirname = 'TFI_dtwa'

    variance_SN_vs_h_vs_range_vs_N = defaultdict(dict)
    t_vs_h_vs_range_vs_N = defaultdict(dict)
    method = 'TFI'

    for Jz in [-1.0]:
        for N in [10,20,50,100,200,500,1000]:
            variance_SN_vs_h_vs_range = defaultdict(dict)
            t_vs_h_vs_range = defaultdict(dict)
            for interaction_range in [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4,4.5,5,5.5,6]:
                fig = plt.figure()
                plt.title(r'TFI, power_law, $\alpha = {}$, $N = {}$, $Jz = {}$'.format(interaction_range, N, Jz))
                plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                plt.xlabel(r'$t$')
                plt.ylim(bottom=0., top=1.)
                variance_SN_vs_h = {}
                t_vs_h = {}
                h_list = sorted(np.concatenate((N * Jz * np.arange(-2,2.25,0.25), np.arange(-2,2.25,0.25))))
                color_idx = np.linspace(1. / len(h_list), 1., len(h_list))
                for i, h in zip(color_idx, h_list): 

                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method, N, 'power_law', 'exp', interaction_range, Jz, h)
                    if filename in os.listdir(dirname): 
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))

                        variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']

                        variance_SN_vs_h[h] = variance_SN_t
                        t_vs_h[h] = t
                        plt.plot(t, variance_SN_t, label='h = {}'.format(h), color=plt.cm.get_cmap('Reds')(i))
                plt.legend(fontsize='xx-small')
                plt.tight_layout()
                plt.savefig('{}/plots/variance_SN_vs_t_N_{}_power_law_exp_{}_Jz_{}_all_h.png'.format(dirname, N, interaction_range, Jz))
                plt.close()
                variance_SN_vs_h_vs_range[interaction_range] = variance_SN_vs_h
                t_vs_h_vs_range[interaction_range] = t_vs_h
            variance_SN_vs_h_vs_range_vs_N[N] = variance_SN_vs_h_vs_range
            t_vs_h_vs_range_vs_N[N] = t_vs_h_vs_range

        optimal_h = defaultdict(dict)
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            plt.title(r'TFI, power_law, $N = {}$, $Jz = {}$'.format(N, Jz))
            plt.xlabel(r'$h$')
            plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
            plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                min_variance_SN_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    try:
                        idx_first_min_variance_SN = argrelextrema(variance_SN_t, np.less_equal)[0][0]
                        h_list.append(h)
                        min_variance_SN_list.append(variance_SN_t[idx_first_min_variance_SN])
                    except:
                        print('error computing minimum for {}, {}, {}'.format(N, interaction_range, h))
                if len(h_list) > 0:
                    plt.plot(h_list, min_variance_SN_list, marker='o', label=r'\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                    optimal_h[interaction_range][N] = np.array(h_list)[np.argmin(min_variance_SN_list)]
            plt.vlines([N/2, -N/2], 0., 1., label='±N/2', linestyle='dashed')
            plt.legend(fontsize='xx-small')
            plt.tight_layout()
            plt.savefig('{}/plots/min_variance_SN_vs_h_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))
            plt.close()

        optimal_h_scaled = defaultdict(dict)
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            plt.title(r'TFI, power_law, $N = {}$, $Jz = {}$'.format(N, Jz))
            plt.xlabel(r'$h / (N \cdot J_z)$')
            plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
            plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                min_variance_SN_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    try:
                        idx_first_min_variance_SN = argrelextrema(variance_SN_t, np.less_equal)[0][0]
                        h_list.append(h)
                        min_variance_SN_list.append(variance_SN_t[idx_first_min_variance_SN])
                    except:
                        print('error computing minimum for {}, {}, {}'.format(N, interaction_range, h))
                if len(h_list) > 0:
                    plt.plot(np.array(h_list) / (N * Jz), min_variance_SN_list, marker='o', label=r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                    optimal_h_scaled[interaction_range][N] = np.array((np.array(h_list) / (N * Jz))[np.argmin(min_variance_SN_list)])
            plt.vlines([1/2, -1/2], 0., 1., label='±1/2', linestyle='dashed')
            plt.legend(fontsize='xx-small')
            plt.tight_layout()
            plt.savefig('{}/plots/min_variance_SN_vs_h_scaled_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))
            plt.close()

        fig = plt.figure(figsize=(7.2,4.8))
        plt.title(r'TFI, power_law, $Jz = {}$'.format(Jz))
        plt.xlabel(r'$N$')
        plt.ylabel(r'$h$')
        for interaction_range, opt_h_vs_N in optimal_h.items():
            N_list = []
            opt_h_list = []
            for N, opt_h in opt_h_vs_N.items():
                N_list.append(N)
                opt_h_list.append(opt_h)
            plt.plot(N_list, opt_h_list, 'o', label=r'$\alpha = {}$'.format(interaction_range))
        plt.legend(fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('{}/plots/opt_h_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        fig = plt.figure(figsize=(7.2,4.8))
        plt.title(r'TFI, power_law, $Jz = {}$'.format(Jz))
        plt.xlabel(r'$N$')
        plt.ylabel(r'$h / (N \cdot J_z)$')
        for interaction_range, opt_h_scaled_vs_N in optimal_h_scaled.items():
            N_list = []
            opt_h_scaled_list = []
            for N, opt_h_scaled in opt_h_scaled_vs_N.items():
                N_list.append(N)
                opt_h_scaled_list.append(opt_h_scaled)
            plt.plot(N_list, opt_h_scaled_list, 'o', label=r'$\alpha = {}$'.format(interaction_range))
        plt.legend(fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('{}/plots/opt_h_scaled_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        optimal_h_for_rate = defaultdict(dict)
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            plt.title(r'TFI, power_law, $N = {}$, $Jz = {}$'.format(N, Jz))
            plt.xlabel(r'$h$')
            plt.ylabel(r'$- \min \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / t_{opt} $')
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                init_rate_variance_SN_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    try:
                        t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                        idx_first_min_variance_SN = argrelextrema(variance_SN_t, np.less_equal)[0][0]                    
                        h_list.append(h)
                        init_rate_variance_SN_list.append((1. - variance_SN_t[idx_first_min_variance_SN]) / t[idx_first_min_variance_SN])
                        # init_rate_variance_SN_list.append((variance_SN_t[0] - variance_SN_t[10]) / (t[10]-t[0]))
                    except:
                        print('error computing minimum for {}, {}, {}'.format(N, interaction_range, h))
                print(h_list, init_rate_variance_SN_list)
                if len(h_list) > 0:
                    plt.plot(h_list, init_rate_variance_SN_list, marker='o', label=r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                    optimal_h_for_rate[interaction_range][N] = np.array(h_list)[np.argmax(init_rate_variance_SN_list)]
            plt.vlines([N/2, -N/2], 0., 1., label='±N/2', linestyle='dashed')
            plt.legend(fontsize='xx-small')
            plt.tight_layout()
            plt.savefig('{}/plots/init_rate_variance_SN_vs_h_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))
            plt.close()


        optimal_h_scaled_for_rate = defaultdict(dict)
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            plt.title(r'TFI, power_law, $N = {}$, $Jz = {}$'.format(N, Jz))
            plt.xlabel(r'$h / (N \cdot J_z)$')
            plt.ylabel(r'$- \min \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / t_{opt} $')
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                init_rate_variance_SN_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    try:
                        t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                        # init_rate_variance_SN_list.append((variance_SN_t[0] - variance_SN_t[10]) / (t[10]-t[0]))
                        idx_first_min_variance_SN = argrelextrema(variance_SN_t, np.less_equal)[0][0]                    
                        h_list.append(h)
                        init_rate_variance_SN_list.append((1. - variance_SN_t[idx_first_min_variance_SN]) / t[idx_first_min_variance_SN])
                    except:
                        print('error computing minimum for {}, {}, {}'.format(N, interaction_range, h))
                if len(h_list) > 0:
                    plt.plot(np.array(h_list) / (N * Jz), init_rate_variance_SN_list, marker='o', label=r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                    optimal_h_scaled_for_rate[interaction_range][N] = np.array((np.array(h_list) / (N * Jz))[np.argmax(init_rate_variance_SN_list)])
            plt.vlines([1/2, -1/2], 0., 1., label='±1/2', linestyle='dashed')
            plt.legend(fontsize='xx-small')
            plt.tight_layout()
            plt.savefig('{}/plots/init_rate_variance_SN_vs_h_scaled_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))
            plt.close()

        fig = plt.figure(figsize=(7.2,4.8))
        plt.title(r'TFI, power_law, $Jz = {}$'.format(Jz))
        plt.xlabel(r'$N$')
        plt.ylabel(r'$h$')
        for interaction_range, opt_h_vs_N in optimal_h_for_rate.items():
            N_list = []
            opt_h_list = []
            for N, opt_h in opt_h_vs_N.items():
                N_list.append(N)
                opt_h_list.append(opt_h)
            plt.plot(N_list, opt_h_list, 'o', label=r'$\alpha = {}$'.format(interaction_range))
        plt.legend(fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('{}/plots/opt_h_for_rate_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        fig = plt.figure(figsize=(7.2,4.8))
        plt.title(r'TFI, power_law, $Jz = {}$'.format(Jz))
        plt.xlabel(r'$N$')
        plt.ylabel(r'$h / (N \cdot J_z)$')
        for interaction_range, opt_h_scaled_vs_N in optimal_h_scaled_for_rate.items():
            N_list = []
            opt_h_scaled_list = []
            for N, opt_h_scaled in opt_h_scaled_vs_N.items():
                N_list.append(N)
                opt_h_scaled_list.append(opt_h_scaled)
            plt.plot(N_list, opt_h_scaled_list, 'o', label=r'$\alpha = {}$'.format(interaction_range))
        plt.legend(fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('{}/plots/opt_h_scaled_for_rate_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        min_variance_SN_vs_N_vs_interaction_range = defaultdict(dict)
        t_opt_vs_N_vs_interaction_range = defaultdict(dict)
        for interaction_range, optimal_h_vs_N in optimal_h_for_rate.items():
            for N, opt_h in optimal_h_vs_N.items():
                variance_SN_t = variance_SN_vs_h_vs_range_vs_N[N][interaction_range][opt_h]
                t = t_vs_h_vs_range_vs_N[N][interaction_range][opt_h]
                idx_first_min_variance_SN = argrelextrema(variance_SN_t, np.less_equal)[0][0]                    
                min_variance_SN_vs_N_vs_interaction_range[interaction_range][N] = variance_SN_t[idx_first_min_variance_SN]
                t_opt_vs_N_vs_interaction_range[interaction_range][N] = t[idx_first_min_variance_SN]

        def fn(x, a, b):
            return a * np.power(x, b)

        fig = plt.figure()
        plt.title(r'TFI, power_law, $Jz = {}$, $h = N \cdot J_z / 2$'.format(Jz))
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
        plt.xlabel(r'$N$')
        plt.ylim(bottom=0., top=1.)
        color_idx = np.linspace(1. / len(min_variance_SN_vs_N_vs_interaction_range), 1., len(min_variance_SN_vs_N_vs_interaction_range))
        b_vs_range = {}
        for i, (interaction_range, min_variance_SN_vs_N) in zip(color_idx, min_variance_SN_vs_N_vs_interaction_range.items()):
            N_list = []
            min_variance_SN_list = []
            for N, min_variance_SN in min_variance_SN_vs_N.items():
                N_list.append(N)
                min_variance_SN_list.append(min_variance_SN)
            prev = plt.plot(N_list, min_variance_SN_list, label=r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
            popt0, pcov0 = curve_fit(fn, np.array(N_list), min_variance_SN_list)
            b_vs_range[interaction_range] = popt0[1]
            plt.plot(N_list, fn(np.array(N_list), *popt0), linestyle='dashed', label=r'$%5.3f * N ^ %5.3f$' % tuple(popt0), color=prev[-1].get_color())
        plt.legend(fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('{}/plots/min_variance_SN_vs_N_Jz_{}_h_opt_all_ranges.png'.format(dirname, Jz))
        plt.yscale('log')
        plt.ylim(bottom=None, top=None)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('{}/plots/log_min_variance_SN_vs_log_N_Jz_{}_h_opt_all_ranges.png'.format(dirname, Jz))
        plt.close()

        fig = plt.figure()
        plt.title(r'TFI, power_law, $Jz = {}$, '.format(Jz) + r'$h = N \cdot J_z / 2$, $\min(N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2) = A * N ^ \beta$')
        plt.ylabel(r'$\beta$')
        plt.xlabel(r'$\alpha$')
        range_list, b_list = [], []
        for interaction_range, b in b_vs_range.items():
            range_list.append(interaction_range)
            b_list.append(-b)
        plt.plot(range_list, b_list, 'o')
        # popt0, pcov0 = curve_fit(fn, np.array(range_list)[2:], b_list[2:])
        # plt.plot(range_list, fn(np.array(range_list), *popt0), linestyle='dashed', label=r'$%5.3f * \alpha ^ %5.3f$' % tuple(popt0))
        plt.legend(fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('{}/plots/b_vs_range_for_min_variance_SN.png'.format(dirname))
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('{}/plots/log_b_vs_log_range_for_min_variance_SN.png'.format(dirname))
        plt.close()


        fig = plt.figure()
        plt.title(r'TFI, power_law, $Jz = {}$, $h = N \cdot J_z / 2$'.format(Jz))
        plt.ylabel(r'$t$')
        plt.xlabel(r'$N$')
        color_idx = np.linspace(1. / len(t_opt_vs_N_vs_interaction_range), 1., len(t_opt_vs_N_vs_interaction_range))
        # m_vs_range = {}
        for i, (interaction_range, t_opt_vs_N) in zip(color_idx, t_opt_vs_N_vs_interaction_range.items()):
            N_list = []
            t_opt_list = []
            for N, t_opt in t_opt_vs_N.items():
                N_list.append(N)
                t_opt_list.append(t_opt)
            prev = plt.plot(N_list, t_opt_list, 'o', label=r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
            # popt0, pcov0 = curve_fit(fn, np.array(N_list), t_opt_list)
            # m_vs_range[interaction_range] = popt0[1]
            # plt.plot(N_list, fn(np.array(N_list), *popt0), linestyle='dashed', label=r'$%5.3f * N ^ %5.3f$' % tuple(popt0), color=prev[-1].get_color())
        plt.legend(fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('{}/plots/t_opt_vs_N_Jz_{}_h_opt_all_ranges.png'.format(dirname, Jz))
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('{}/plots/log_t_opt_vs_log_N_Jz_{}_h_opt_all_ranges.png'.format(dirname, Jz))
        plt.close()

        # fig = plt.figure()
        # plt.title(r'TFI, power_law, $Jz = {}$, '.format(Jz) + r'$h = N \cdot J_z / 2$, $t_{opt} = A * N ^ \mu$')
        # plt.ylabel(r'$-\mu$')
        # plt.xlabel(r'$\alpha$')
        # range_list, m_list = [], []
        # for interaction_range, m in m_vs_range.items():
        #     range_list.append(interaction_range)
        #     m_list.append(-m)
        # plt.plot(range_list, m_list)
        # popt0, pcov0 = curve_fit(fn, np.array(range_list)[2:], b_list[2:])
        # plt.plot(range_list, fn(np.array(range_list), *popt0), linestyle='dashed', label=r'$%5.3f * \alpha ^ %5.3f$' % tuple(popt0))
        # plt.legend(fontsize='xx-small')
        # plt.tight_layout()
        # plt.savefig('{}/plots/m_vs_range_for_t_opt.png'.format(dirname))
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.tight_layout()
        # plt.savefig('{}/plots/log_m_vs_log_range_for_t_opt.png'.format(dirname))
        # plt.close()

        # fig = plt.figure()
        # plt.title(r'TFI, power_law, $Jz = {}$, '.format(Jz) + r'$h = N \cdot J_z / 2$, $t_{opt} = A * N ^ \mu$')
        # plt.ylabel(r'$- \min \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / t_{opt} $')
        # plt.xlabel(r'$N$')
        # color_idx = np.linspace(1. / len(optimal_h_for_rate), 1., len(optimal_h_for_rate))
        # for i, (interaction_range, optimal_h_vs_N) in zip(color_idx, optimal_h_for_rate.items()):
        #     N_list = []
        #     init_rate_variance_SN_list = []
        #     for N, opt_h in optimal_h_vs_N.items():
        #         variance_SN_t = variance_SN_vs_h_vs_range_vs_N[N][interaction_range][opt_h]
        #         t = t_vs_h_vs_range_vs_N[N][interaction_range][opt_h]
        #         N_list.append(N)
        #         init_rate_variance_SN_list.append((variance_SN_t[1]-variance_SN_t[0]) / (t[1]-t[0]))
        #     plt.plot(N_list, init_rate_variance_SN_list, label=r'$\alpha = {}$'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        # plt.legend(fontsize='xx-small')
        # plt.tight_layout()
        # plt.savefig('{}/plots/init_rate_variance_SN_vs_N_Jz_{}_h_opt_rate_all_ranges.png'.format(dirname, Jz))
        # plt.close()
  

  

