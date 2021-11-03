import numpy as np
import setup
import spin_dynamics as sd
import util
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import os

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    beta_vs_range_vs_method = defaultdict(dict)
    gamma_vs_range_vs_method = defaultdict(dict)
    mu_vs_range_vs_method = defaultdict(dict)
    range_list = [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.]
    N_list = [10,20,50,100,200,500,1000]

    for interaction_range in range_list:
        
        fig = plt.figure(figsize=(8,6))
        plt.title(r'power law, $\alpha = {}$'.format(interaction_range))
        plt.xlabel(r'$t$')
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
        
        color_idx = np.linspace(1. / len(N_list), 1., len(N_list))
        method_list = ['CT', 'CTv2', 'ZZ', 'XY', 'TFI, h=NJ/2', 'XXZ, J_eff=-0.1']
        min_variance_SN_vs_N_vs_method = defaultdict(dict)
        variance_SN_rate_vs_N_vs_method = defaultdict(dict)
        t_opt_vs_N_vs_method = defaultdict(dict)
        for method in method_list:
            
            for i, N in zip(color_idx, N_list):
                    if method == 'CT':
                        dirname = method + '_dtwa'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, 1.)
                        color = plt.cm.get_cmap('Reds')(i)
                        t_scale = 2
                    elif method == 'CTv2':
                        dirname = method + '_dtwa'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, 1.)
                        color = plt.cm.get_cmap('Oranges')(i)
                        t_scale = 3
                    elif method == 'ZZ':
                        dirname = method + '_dtwa'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, 'power_law', 'exp', interaction_range)
                        color = plt.cm.get_cmap('Blues')(i)
                        t_scale = 1
                    elif method == 'XY':
                        dirname = method + '_dtwa'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, 'power_law', 'exp', interaction_range)
                        color = plt.cm.get_cmap('Greens')(i)
                        t_scale = 2
                    elif method == 'TFI, h=NJ/2':
                        method_name = 'TFI'
                        dirname = method_name + '_dtwa'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, -1., - N / 2)
                        color = plt.cm.get_cmap('Purples')(i)
                        t_scale = 1
                    elif method == 'XXZ, J_eff=-0.1':
                        J_eff = -0.1
                        method_name = 'XXZ'
                        dirname = method_name + '_dtwa'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
                        color = plt.cm.get_cmap('Greys')(i)
                        Jz = (J_eff + 1.) / 2
                        Jperp = 2 * (1 - Jz)
                        t_scale = Jz + 2 * (Jperp / 2)
                    if filename in os.listdir(dirname):
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                        plt.plot(t, variance_SN_t, label=method + r', $N = {}$'.format(N), color=color)

                        min_variance_SN_vs_N_vs_method[method][N] = min(variance_SN_t)
                        t_opt_vs_N_vs_method[method][N] = t_scale * t[np.argmin(variance_SN_t)]
                        variance_SN_rate_vs_N_vs_method[method][N] = (1. - min(variance_SN_t)) / t[np.argmin(variance_SN_t)]

        plt.ylim(bottom=0., top=1.)
        plt.xlim(left=0., right=2.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('CT_ZZ_TFI_dtwa/variance_SN_vs_t_power_law_exp_{}_all_N.png'.format(interaction_range))
        plt.close()

        def fn(N, a, b):
            return a * np.power(N, b)

        fig = plt.figure(figsize=(8,6))
        plt.title(r'power law, $\alpha = {}$'.format(interaction_range))
        plt.xlabel(r'$N$')
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
        for method, min_variance_SN_vs_N in min_variance_SN_vs_N_vs_method.items():
            Ns = []
            min_variance_SNs = []
            for N, min_variance_SN in min_variance_SN_vs_N.items():
                Ns.append(N)
                min_variance_SNs.append(min_variance_SN)
            prev = plt.plot(Ns, min_variance_SNs, 'o', label=method)
            try:
                popt, pcov = curve_fit(fn, np.array(Ns), min_variance_SNs)
                plt.plot(Ns, fn(np.array(Ns), *popt), label=r'$%5.3f \cdot N^{%5.3f}$' % tuple(popt), linestyle='dashed', color=prev[-1].get_color())
                beta_vs_range_vs_method[method][interaction_range] = tuple(popt)[1]
            except:
                None
        plt.ylim(bottom=0., top=1.)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
        plt.tight_layout()
        plt.savefig('CT_ZZ_TFI_dtwa/min_variance_SN_vs_N_power_law_exp_{}.png'.format(interaction_range))
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(bottom=0.001, top=1.)
        plt.tight_layout()
        plt.savefig('CT_ZZ_TFI_dtwa/log_min_variance_SN_vs_log_N_power_law_exp_{}.png'.format(interaction_range))
        plt.close()

        fig = plt.figure(figsize=(8,6))
        plt.title(r'power law, $\alpha = {}$'.format(interaction_range))
        plt.xlabel(r'$N$')
        plt.ylabel(r'$t_{opt}$')
        for method, t_opt_vs_N in t_opt_vs_N_vs_method.items():
            Ns = []
            t_opts = []
            for N, t_opt in t_opt_vs_N.items():
                Ns.append(N)
                t_opts.append(t_opt)
            prev = plt.plot(Ns, t_opts, 'o', label=method)
            try:
                popt, pcov = curve_fit(fn, np.array(Ns), t_opts)
                plt.plot(Ns, fn(np.array(Ns), *popt), label=r'$%5.3f \cdot N^{%5.3f}$' % tuple(popt), linestyle='dashed', color=prev[-1].get_color())
                mu_vs_range_vs_method[method][interaction_range] = tuple(popt)[1]
            except:
                None
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
        plt.tight_layout()
        plt.savefig('CT_ZZ_TFI_dtwa/t_opt_vs_N_power_law_exp_{}.png'.format(interaction_range))
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('CT_ZZ_TFI_dtwa/log_t_opt_vs_log_N_power_law_exp_{}.png'.format(interaction_range))
        plt.close()

        fig = plt.figure(figsize=(8,6))
        plt.title(r'power law, $\alpha = {}$'.format(interaction_range))
        plt.xlabel(r'$N$')
        plt.ylabel(r'$- \min \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / t_{opt} $')
        for method, variance_SN_rate_vs_N in variance_SN_rate_vs_N_vs_method.items():
            Ns = []
            variance_SN_rates = []
            for N, variance_SN_rate in variance_SN_rate_vs_N.items():
                Ns.append(N)
                variance_SN_rates.append(variance_SN_rate)
            prev = plt.plot(Ns, variance_SN_rates, 'o', label=method)
            try:
                popt, pcov = curve_fit(fn, np.array(Ns), variance_SN_rates)
                plt.plot(Ns, fn(np.array(Ns), *popt), label=r'$%5.3f \cdot N^{%5.3f}$' % tuple(popt), linestyle='dashed', color=prev[-1].get_color())
                gamma_vs_range_vs_method[method][interaction_range] = tuple(popt)[1]
            except:
                None
        # plt.ylim(bottom=0., top=1.)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
        plt.tight_layout()
        plt.savefig('CT_ZZ_TFI_dtwa/variance_SN_rate_vs_N_power_law_exp_{}.png'.format(interaction_range))
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('CT_ZZ_TFI_dtwa/log_variance_SN_rate_vs_log_N_power_law_exp_{}.png'.format(interaction_range))
        plt.close()

    def fn2(N, a, b, c, d):
        return a / (b + np.power(N, c)) + d
    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $\min(N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2) = A * N ^ \beta$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$- \beta$')
    for method, beta_vs_range in beta_vs_range_vs_method.items():
        betas = []
        ranges = []
        for interaction_range, beta in beta_vs_range.items():
            betas.append(- beta)
            ranges.append(interaction_range)
        prev = plt.plot(ranges, betas, 'o', label=method)
        popt0, pcov0 = curve_fit(fn2, np.array(ranges), betas)
        plt.plot(np.logspace(np.log10(0.5), np.log10(6), num=1000), fn2(np.logspace(np.log10(0.5), np.log10(6), num=1000), *popt0), color=prev[-1].get_color(), linestyle='dashed', label=r'$%5.3f / (%5.3f + \alpha^{%5.3f}) + %5.3f$' % tuple(popt0))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    # plt.ylim(0.001, 1)
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_dtwa/b_vs_range_for_min_variance_SN.png')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('CT_ZZ_TFI_dtwa/log_b_vs_log_range_for_min_variance_SN.png')
    plt.close()

    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $t_{opt} = A * N ^ \mu$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$- \mu$')
    for method, mu_vs_range in mu_vs_range_vs_method.items():
        mus = []
        ranges = []
        for interaction_range, mu in mu_vs_range.items():
            mus.append(- mu)
            ranges.append(interaction_range)
        plt.plot(ranges, mus, 'o', label=method)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    # plt.ylim(0.001, 1)
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_dtwa/m_vs_range_for_t_opt.png')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('CT_ZZ_TFI_dtwa/log_m_vs_log_range_for_t_opt.png')
    plt.close()

    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $- \min \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / t_{opt} = A * N ^ \gamma$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\gamma$')
    for method, gamma_vs_range in gamma_vs_range_vs_method.items():
        gammas = []
        ranges = []
        for interaction_range, gamma in gamma_vs_range.items():
            gammas.append(gamma)
            ranges.append(interaction_range)
        prev = plt.plot(ranges, gammas, 'o', label=method)
        # popt0, pcov0 = curve_fit(fn2, np.array(ranges), gammas)
        # plt.plot(np.logspace(np.log(0.5), np.log(6), num=1000), fn2(np.logspace(np.log(0.5), np.log(6), num=1000), *popt0), color=prev[-1].get_color(), linestyle='dashed', label='%5.3f / (%5.3f + exp ^ %5.3f) + %5.3f' % tuple(popt0))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.tight_layout()
    # plt.ylim(0.001, 1)
    plt.savefig('CT_ZZ_TFI_dtwa/g_vs_range_for_variance_SN_rate.png')


