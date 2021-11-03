import numpy as np
from scipy.optimize import curve_fit
import setup
import spin_dynamics as sd
import util
from collections import defaultdict
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    dirname = 'XXZ_dtwa_bootstrap'

    variance_SN_vs_J_eff_vs_range_vs_N = defaultdict(dict)
    t_vs_J_eff_vs_range_vs_N = defaultdict(dict)
    variance_SN_vs_J_eff_vs_N_vs_range = defaultdict(dict)
    t_vs_J_eff_vs_N_vs_range = defaultdict(dict)
    method = 'XXZ'
    Hlim_vs_N_vs_range = defaultdict(list)
    for N in [10,20,50,100,200]:
    # for N in [100]:
        variance_SN_vs_J_eff_vs_range = defaultdict(dict)
        t_vs_J_eff_vs_range = defaultdict(dict)
        for interaction_range in [0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
            variance_SN_vs_J_eff = {}
            t_vs_J_eff = {}
            J_eff_list_full = sorted([1., -1., -0.02, -0.04, -0.06, -0.08, -0.1, -0.12, -0.14, -0.16, -0.18, -0.2, -0.22, -0.24, -0.26, -0.28, -0.3, -0.32, -0.34, -0.36, -0.38, -0.4, 0.01, 0.005, 0.0025, 0.00125, -0.01, -0.005, -0.0025, -0.00125])
            # J_eff_list = sorted(np.concatenate(([1, -1], -1 * np.linspace(0.02, 0.40, 19, endpoint=False), 0.01 * (2 ** (- np.linspace(0, 3, 4))), - 0.01 * (2 ** (- np.linspace(0, 3, 4))))))
            Hlims = []
            for J_eff in J_eff_list_full:
                variance_SN_t_samples = []
                t_samples = []
                for bs in range(50):

                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J_eff, bs)
                    if os.path.isfile('{}/{}'.format(dirname, filename)):
                        try:
                            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                            variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                            variance_SN_t_samples.append(variance_SN_t)
                            t_samples.append(t)
                        except:
                            None
                if len(variance_SN_t_samples) > 0:
                    variance_SN_vs_J_eff[J_eff] = variance_SN_t_samples
                    t_vs_J_eff[J_eff] = t_samples[0]
                    Hlims.append(np.mean([min(variance_SN_t) for variance_SN_t in variance_SN_t_samples]))

            variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range] = variance_SN_vs_J_eff
            t_vs_J_eff_vs_range_vs_N[N][interaction_range] = t_vs_J_eff
            variance_SN_vs_J_eff_vs_N_vs_range[interaction_range][N] = variance_SN_vs_J_eff
            t_vs_J_eff_vs_N_vs_range[interaction_range][N] = t_vs_J_eff

            Hlim = min(Hlims) if len(Hlims) > 0 else np.nan
            Hlim_vs_N_vs_range[interaction_range].append(Hlim)

    def fn0(N, a, b):
        return a / np.power(N, b)

    fig = plt.figure()
    title = 'power law'
    plt.title(title)
    plt.xlabel('N')
    plt.ylabel('N * <S_a^2> / <S_x>^2')
    N_list = np.array([10,20,50,100,200])
    for interaction_range, Hlim_vs_N in Hlim_vs_N_vs_range.items():
        if interaction_range == 0.:
            continue
        plt.plot(N_list, Hlim_vs_N, label='exp = {}'.format(interaction_range))
        popt0, pcov0 = curve_fit(fn0, N_list, Hlim_vs_N)
        plt.plot(N_list, fn0(N_list, *popt0), linestyle='dashed', label='{} / N ^ {}'.format(*popt0))


    plt.plot([10,20,50,100,200], 1. / np.array([10,20,50,100,200]), linestyle='dashed', label='1 / N')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/min_variance_SN_vs_N_all_ranges.png')
    plt.close()

    def fn1(J_eff, a, b, c):
        return a + b * np.power(np.abs(J_eff), c)

    params_vs_range_vs_N = defaultdict(dict)
    params_vs_N_vs_range = defaultdict(dict)
    for N, variance_SN_vs_J_eff_vs_range in variance_SN_vs_J_eff_vs_range_vs_N.items():

        fig = plt.figure(figsize=(7.2,7.2))
        title = 'N = {}, power law'.format(N)
        plt.title(title)
        plt.xlabel('J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        plt.ylim(bottom=0., top=1.)
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_range), 1., len(variance_SN_vs_J_eff_vs_range))

        popt1_vs_r = {}
        popt2_vs_r = {}

        for i, (interaction_range, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_range.items()):
            J_eff_list = []
            min_variance_SN_list = []
            min_variance_SN_err_list = []
            for J_eff, variance_SN_t_samples in variance_SN_vs_J_eff.items():
                min_variance_SN_samples = [min(variance_SN_t) for variance_SN_t in variance_SN_t_samples]
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    min_variance_SN_list.append(np.mean(min_variance_SN_samples))
                    min_variance_SN_err_list.append(np.std(min_variance_SN_samples, ddof=1))
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(np.mean(min_variance_SN_samples), -0.4, 0.001, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            
            offset = 1.1 * (1. / np.power(N, 2/3))
            def fn2(J_eff, b, c):
                return 1.1 * (1. / np.power(N, 2/3)) + b * np.power(np.abs(J_eff), c)

            plt.errorbar(J_eff_list, np.array(min_variance_SN_list), yerr=min_variance_SN_err_list, marker='.', linestyle='None', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))

            # x = [J for J in J_eff_list if np.abs(J) <= 0.02]
            # y = [min_variance_SN_list[i] for i in range(len(min_variance_SN_list)) if np.abs(J_eff_list[i]) <= 0.02]
            x = [J for J in J_eff_list if J < 0.]
            y = [min_variance_SN_list[i] for i in range(len(min_variance_SN_list)) if J_eff_list[i] < 0.]

            try:
                popt1, pcov1 = curve_fit(fn1, np.array(x), np.array(y))
                popt1_vs_r[interaction_range] = popt1
                params_vs_range_vs_N[N][interaction_range] = (tuple(popt1), np.sqrt(np.diag(pcov1)))
                params_vs_N_vs_range[interaction_range][N] = (tuple(popt1), np.sqrt(np.diag(pcov1)))
                plt.plot(J_eff_list, fn1(np.array(J_eff_list), *popt1), linestyle='dashed', color=plt.cm.get_cmap('Reds')(i), label='%5.3f + %5.3f * |J_eff| ^ %5.3f' % tuple(popt1))
            except:
                None
            try:
                popt2, pcov2 = curve_fit(fn2, np.array(x), np.array(y))
                popt2_vs_r[interaction_range] = popt2
                plt.plot(J_eff_list, fn2(np.array(J_eff_list), *popt2), linestyle='dotted', color=plt.cm.get_cmap('Reds')(i), label='1.1 / N ^ (2/3) + %5.3f * |J_eff| ^ %5.3f' % tuple(popt2))
            except:
                None

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/min_variance_SN_vs_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

        fig = plt.figure(figsize=(7.2,7.2))
        title = 'N = {}, power law'.format(N)
        plt.title(title)
        plt.xlabel('- J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_range), 1., len(variance_SN_vs_J_eff_vs_range))
        for i, (interaction_range, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_range.items()):
            J_eff_list = []
            min_variance_SN_list = []
            min_variance_SN_err_list = []
            for J_eff, variance_SN_t_samples in variance_SN_vs_J_eff.items():
                min_variance_SN_samples = [min(variance_SN_t) for variance_SN_t in variance_SN_t_samples]
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        min_variance_SN_list.append(np.mean(min_variance_SN_samples))
                        min_variance_SN_err_list.append(np.std(min_variance_SN_samples, ddof=1))
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(np.mean(min_variance_SN_samples), 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))

            offset = 1.1 * (1. / np.power(N, 2/3))
            def fn2(J_eff, b, c):
                return 1.1 * (1. / np.power(N, 2/3)) + b * np.power(np.abs(J_eff), c)

            plt.errorbar(- np.array(J_eff_list), np.array(min_variance_SN_list), yerr=min_variance_SN_err_list, marker='.', linestyle='None', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        

            try:
                popt1 = popt1_vs_r[interaction_range]
                plt.plot(- np.array(J_eff_list), fn1(np.array(J_eff_list), *popt1), linestyle='dashed', color=plt.cm.get_cmap('Reds')(i), label='%5.3f + %5.3f * |J_eff| ^ %5.3f' % tuple(popt1))
            except:
                None
            try:
                popt2 = popt2_vs_r[interaction_range]
                plt.plot(- np.array(J_eff_list), fn2(np.array(J_eff_list), *popt2), linestyle='dotted', color=plt.cm.get_cmap('Reds')(i), label='1.1 / N ^ (2/3) + %5.3f * |J_eff| ^ %5.3f' % tuple(popt2))
            except:
                None

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/log_min_variance_SN_vs_log_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

        fig = plt.figure(figsize=(7.2,7.2))
        title = 'N = {}, power law'.format(N)
        plt.title(title)
        plt.xlabel('J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        plt.ylim(bottom=0., top=1.)
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_range), 1., len(variance_SN_vs_J_eff_vs_range))

        for i, (interaction_range, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_range.items()):
            J_eff_list = []
            min_variance_SN_list = []
            min_variance_SN_err_list = []
            for J_eff, variance_SN_t_samples in variance_SN_vs_J_eff.items():
                min_variance_SN_samples = [min(variance_SN_t) for variance_SN_t in variance_SN_t_samples]
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    min_variance_SN_list.append(np.mean(min_variance_SN_samples))
                    min_variance_SN_err_list.append(np.std(min_variance_SN_samples, ddof=1))
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(np.mean(min_variance_SN_samples), -0.4, 0.001, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            
            offset = 1.1 * (1. / np.power(N, 2/3))
            def fn2(J_eff, b, c):
                return 1.1 * (1. / np.power(N, 2/3)) + b * np.power(np.abs(J_eff), c)
            plt.errorbar(J_eff_list, np.array(min_variance_SN_list) - offset, yerr=min_variance_SN_err_list, marker='.', linestyle='None', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))

            # x = [J for J in J_eff_list if np.abs(J) <= 0.02]
            # y = [min_variance_SN_list[i] for i in range(len(min_variance_SN_list)) if np.abs(J_eff_list[i]) <= 0.02]
            x = [J for J in J_eff_list if J < 0.]
            y = [min_variance_SN_list[i] for i in range(len(min_variance_SN_list)) if J_eff_list[i] < 0.]

            try:
                popt1 = popt1_vs_r[interaction_range]
                plt.plot(J_eff_list, fn1(np.array(J_eff_list), *popt1) - offset, linestyle='dashed', color=plt.cm.get_cmap('Reds')(i), label='%5.3f + %5.3f * |J_eff| ^ %5.3f' % tuple(popt1))
            except:
                None
            try:
                popt2 = popt2_vs_r[interaction_range]
                plt.plot(J_eff_list, fn2(np.array(J_eff_list), *popt2) - offset, linestyle='dotted', color=plt.cm.get_cmap('Reds')(i), label='1.1 / N ^ (2/3) + %5.3f * |J_eff| ^ %5.3f' % tuple(popt2))
            except:
                None

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/min_variance_SN_offset_vs_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

        fig = plt.figure(figsize=(7.2,7.2))
        title = 'N = {}, power law'.format(N)
        plt.title(title)
        plt.xlabel('- J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_range), 1., len(variance_SN_vs_J_eff_vs_range))
        for i, (interaction_range, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_range.items()):
            J_eff_list = []
            min_variance_SN_list = []
            min_variance_SN_err_list = []
            for J_eff, variance_SN_t_samples in variance_SN_vs_J_eff.items():
                min_variance_SN_samples = [min(variance_SN_t) for variance_SN_t in variance_SN_t_samples]
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        min_variance_SN_list.append(np.mean(min_variance_SN_samples))
                        min_variance_SN_err_list.append(np.std(min_variance_SN_samples, ddof=1))
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(np.mean(min_variance_SN_samples), 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            offset = 1.1 * (1. / np.power(N, 2/3))
            def fn2(J_eff, b, c):
                return 1.1 * (1. / np.power(N, 2/3)) + b * np.power(np.abs(J_eff), c)
            
            plt.errorbar(- np.array(J_eff_list), np.array(min_variance_SN_list) - offset, yerr=min_variance_SN_err_list, marker='.', linestyle='None', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))

            try:
                popt1 = popt1_vs_r[interaction_range]
                plt.plot(- np.array(J_eff_list), fn1(np.array(J_eff_list), *popt1) - offset, linestyle='dashed', color=plt.cm.get_cmap('Reds')(i), label='%5.3f + %5.3f * |J_eff| ^ %5.3f' % tuple(popt1))
            except:
                None
            try:
                popt2 = popt2_vs_r[interaction_range]
                plt.plot(- np.array(J_eff_list), fn2(np.array(J_eff_list), *popt2) - offset, linestyle='dotted', color=plt.cm.get_cmap('Reds')(i), label='1.1 / N ^ (2/3) + %5.3f * |J_eff| ^ %5.3f' % tuple(popt2))
            except:
                None

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/log_min_variance_SN_offset_vs_log_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

    params2_vs_range_vs_N = defaultdict(dict)
    params2_vs_N_vs_range = defaultdict(dict)
    for interaction_range, variance_SN_vs_J_eff_vs_N in variance_SN_vs_J_eff_vs_N_vs_range.items():
        fig = plt.figure(figsize=(7.2,7.2))
        title = 'exp = {}, power law'.format(interaction_range)
        plt.title(title)
        plt.xlabel('J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        plt.ylim(bottom=0., top=1.)
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_N), 1., len(variance_SN_vs_J_eff_vs_N))

        popt1_vs_N = {}
        popt2_vs_N = {}

        for i, (N, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_N.items()):
            J_eff_list = []
            min_variance_SN_list = []
            min_variance_SN_err_list = []
            for J_eff, variance_SN_t_samples in variance_SN_vs_J_eff.items():
                min_variance_SN_samples = [min(variance_SN_t) for variance_SN_t in variance_SN_t_samples]
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    min_variance_SN_list.append(np.mean(min_variance_SN_samples))
                    min_variance_SN_err_list.append(np.std(min_variance_SN_samples, ddof=1))
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(np.mean(min_variance_SN_samples), -0.4, 0.01, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('N', N, J_eff))
            
            def fn2(J_eff, b, c):
                return 1.1 * (1. / np.power(N, 2/3)) + b * np.power(np.abs(J_eff), c)

            plt.errorbar(J_eff_list, min_variance_SN_list, yerr=min_variance_SN_err_list, marker='.', linestyle='None', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))

            # x = [J_eff_list[i] for i in range(len(J_eff_list)) if min_variance_SN_list[i] <= 1.2 * min(min_variance_SN_list)]
            # y = [min_variance_SN_list[i] for i in range(len(min_variance_SN_list)) if min_variance_SN_list[i] <= 1.2 * min(min_variance_SN_list)]
            x = J_eff_list
            y = min_variance_SN_list

            try:
                popt1, pcov1 = curve_fit(fn1, np.array(x), np.array(y))
                popt1_vs_N[N] = popt1
                params2_vs_range_vs_N[N][interaction_range] = (tuple(popt1), np.sqrt(np.diag(pcov1)))
                params2_vs_N_vs_range[interaction_range][N] = (tuple(popt1), np.sqrt(np.diag(pcov1)))
                
                plt.plot(J_eff_list, fn1(np.array(J_eff_list), *popt1), linestyle='dashed', color=plt.cm.get_cmap('Reds')(i), label='%5.3f + %5.3f * |J_eff| ^ %5.3f' % tuple(popt1))
            except:
                None
            try:
                popt2, pcov2 = curve_fit(fn2, np.array(x), np.array(y))
                popt2_vs_N[N] = popt2
                plt.plot(J_eff_list, fn2(np.array(J_eff_list), *popt2), linestyle='dotted', color=plt.cm.get_cmap('Reds')(i), label='1.1 / N ^ (2/3) + %5.3f * |J_eff| ^ %5.3f' % tuple(popt2))
            except:
                None

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/min_variance_SN_vs_J_eff_range_{}_all_N.png'.format(interaction_range))
        plt.close()

        fig = plt.figure(figsize=(7.2,7.2))
        title = 'exp = {}, power law'.format(interaction_range)
        plt.title(title)
        plt.xlabel('- J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_N), 1., len(variance_SN_vs_J_eff_vs_N))
        for i, (N, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_N.items()):
            J_eff_list = []
            min_variance_SN_list = []
            min_variance_SN_err_list = []
            for J_eff, variance_SN_t_samples in variance_SN_vs_J_eff.items():
                min_variance_SN_samples = [min(variance_SN_t) for variance_SN_t in variance_SN_t_samples]
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        min_variance_SN_list.append(np.mean(min_variance_SN_samples))
                        min_variance_SN_err_list.append(np.std(min_variance_SN_samples, ddof=1))
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(np.mean(min_variance_SN_samples), 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('N', N, J_eff))
            plt.errorbar(- np.array(J_eff_list), min_variance_SN_list, yerr=min_variance_SN_err_list, marker='.', linestyle='None', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))

            try:
                popt1 = popt1_vs_N[N]
                popt1, pcov1 = curve_fit(fn1, np.array(J_eff_list), min_variance_SN_list)
                plt.plot(- np.array(J_eff_list), fn1(np.array(J_eff_list), *popt1), linestyle='dashed', color=plt.cm.get_cmap('Reds')(i), label='%5.3f + %5.3f * |J_eff| ^ %5.3f' % tuple(popt1))
            except:
                None
            try:
                popt2 = popt2_vs_N[N]
                plt.plot(- np.array(J_eff_list), fn2(np.array(J_eff_list), *popt2), linestyle='dotted', color=plt.cm.get_cmap('Reds')(i), label='1.1 / N ^ (2/3) + %5.3f * |J_eff| ^ %5.3f' % tuple(popt2))
            except:
                None
                
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/log_min_variance_SN_vs_log_J_eff_range_{}_all_N.png'.format(interaction_range))
        plt.close()

    fig = plt.figure()
    plt.title('power_law, a + b * |J_eff| ^ c')
    plt.xlabel('exp')
    color_idx = np.linspace(1. / len(params_vs_range_vs_N), 1., len(params_vs_range_vs_N))
    for i, (N, params_vs_range) in zip(color_idx, params_vs_range_vs_N.items()):
        range_list = []
        a_list = []
        b_list = []
        c_list = []
        a_err_list = []
        b_err_list = []
        c_err_list = []
        for interaction_range, params in params_vs_range.items():
            range_list.append(interaction_range)
            (a, b, c), (a_err, b_err, c_err) = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            a_err_list.append(a_err)
            b_err_list.append(b_err)
            c_err_list.append(c_err)
        plt.errorbar(range_list, a_list, yerr=a_err_list, marker='.', label='a, N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
        # plt.plot(range_list, b_list, label='b, N = {}'.format(N), color=plt.cm.get_cmap('Blues')(i))
        # plt.plot(range_list, c_list, label='c, N = {}'.format(N), color=plt.cm.get_cmap('Greens')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/a_vs_range_all_N.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_a_vs_log_range_all_N.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, a + b * |J_eff| ^ c')
    plt.xlabel('exp')
    color_idx = np.linspace(1. / len(params_vs_range_vs_N), 1., len(params_vs_range_vs_N))
    for i, (N, params_vs_range) in zip(color_idx, params_vs_range_vs_N.items()):
        range_list = []
        a_list = []
        b_list = []
        c_list = []
        a_err_list = []
        b_err_list = []
        c_err_list = []
        for interaction_range, params in params_vs_range.items():
            range_list.append(interaction_range)
            (a, b, c), (a_err, b_err, c_err) = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            a_err_list.append(a_err)
            b_err_list.append(b_err)
            c_err_list.append(c_err)
        plt.errorbar(range_list, b_list, yerr=b_err_list, marker='.', label='b, N = {}'.format(N), color=plt.cm.get_cmap('Blues')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/b_vs_range_all_N.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_b_vs_log_range_all_N.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, a + b * |J_eff| ^ c')
    plt.xlabel('exp')
    color_idx = np.linspace(1. / len(params_vs_range_vs_N), 1., len(params_vs_range_vs_N))
    for i, (N, params_vs_range) in zip(color_idx, params_vs_range_vs_N.items()):
        range_list = []
        a_list = []
        b_list = []
        c_list = []
        a_err_list = []
        b_err_list = []
        c_err_list = []
        for interaction_range, params in params_vs_range.items():
            range_list.append(interaction_range)
            (a, b, c), (a_err, b_err, c_err) = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            a_err_list.append(a_err)
            b_err_list.append(b_err)
            c_err_list.append(c_err)
        # plt.plot(range_list, a_list, label='a, N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
        # plt.plot(range_list, b_list, label='b, N = {}'.format(N), color=plt.cm.get_cmap('Blues')(i))
        plt.errorbar(range_list, c_list, yerr=c_err_list, marker='.', label='c, N = {}'.format(N), color=plt.cm.get_cmap('Greens')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/c_vs_range_all_N.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_c_vs_log_range_all_N.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, d + e * |J_eff| ^ f')
    plt.xlabel('exp')
    color_idx = np.linspace(1. / len(params2_vs_range_vs_N), 1., len(params2_vs_range_vs_N))
    for i, (N, params_vs_range) in zip(color_idx, params2_vs_range_vs_N.items()):
        range_list = []
        d_list = []
        e_list = []
        f_list = []
        d_err_list = []
        e_err_list = []
        f_err_list = []
        for interaction_range, params in params_vs_range.items():
            range_list.append(interaction_range)
            (d, e, f), (d_err, e_err, f_err) = params
            d_list.append(d)
            e_list.append(d)
            f_list.append(f)
            d_err_list.append(d_err)
            e_err_list.append(e_err)
            f_err_list.append(f_err)
        plt.errorbar(range_list, d_list, yerr=d_err_list, marker='.', label='d, N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/d_vs_range_all_N.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_d_vs_log_range_all_N.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, d + e * |J_eff| ^ f')
    plt.xlabel('exp')
    color_idx = np.linspace(1. / len(params2_vs_range_vs_N), 1., len(params2_vs_range_vs_N))
    for i, (N, params_vs_range) in zip(color_idx, params2_vs_range_vs_N.items()):
        range_list = []
        d_list = []
        e_list = []
        f_list = []
        d_err_list = []
        e_err_list = []
        f_err_list = []
        for interaction_range, params in params_vs_range.items():
            range_list.append(interaction_range)
            (d, e, f), (d_err, e_err, f_err) = params
            d_list.append(d)
            e_list.append(d)
            f_list.append(f)
            d_err_list.append(d_err)
            e_err_list.append(e_err)
            f_err_list.append(f_err)
        plt.errorbar(range_list, e_list, yerr=e_err_list, marker='.', label='e, N = {}'.format(N), color=plt.cm.get_cmap('Blues')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/e_vs_range_all_N.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_e_vs_log_range_all_N.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, d + e * |J_eff| ^ f')
    plt.xlabel('exp')
    color_idx = np.linspace(1. / len(params2_vs_range_vs_N), 1., len(params2_vs_range_vs_N))
    for i, (N, params_vs_range) in zip(color_idx, params2_vs_range_vs_N.items()):
        range_list = []
        d_list = []
        e_list = []
        f_list = []
        d_err_list = []
        e_err_list = []
        f_err_list = []
        for interaction_range, params in params_vs_range.items():
            range_list.append(interaction_range)
            (d, e, f), (d_err, e_err, f_err) = params
            d_list.append(d)
            e_list.append(d)
            f_list.append(f)
            d_err_list.append(d_err)
            e_err_list.append(e_err)
            f_err_list.append(f_err)
        plt.errorbar(range_list, f_list, yerr=f_err_list, marker='.', label='f, N = {}'.format(N), color=plt.cm.get_cmap('Greens')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/f_vs_range_all_N.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_f_vs_log_range_all_N.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, a + b * |J_eff| ^ c')
    plt.xlabel('N')
    color_idx = np.linspace(1. / len(params_vs_N_vs_range), 1., len(params_vs_N_vs_range))
    for i, (interaction_range, params_vs_N) in zip(color_idx, params_vs_N_vs_range.items()):
        N_list = []
        a_list = []
        b_list = []
        c_list = []
        a_err_list = []
        b_err_list = []
        c_err_list = []
        for N, params in params_vs_N.items():
            N_list.append(N)
            (a, b, c), (a_err, b_err, c_err) = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            a_err_list.append(a_err)
            b_err_list.append(b_err)
            c_err_list.append(c_err)
        plt.errorbar(N_list, a_list, yerr=a_err_list, marker='.', label='a, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
    plt.plot(N_list, 1 / np.array(N_list), linestyle='dashed', label='1 / N')
    plt.plot(N_list, 1 / np.power(N_list, 2/3), linestyle='dashed', label='1 / (N ^ (2/3))')
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/a_vs_N_all_range.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_a_vs_log_N_all_range.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, a + b * |J_eff| ^ c')
    plt.xlabel('N')
    color_idx = np.linspace(1. / len(params_vs_N_vs_range), 1., len(params_vs_N_vs_range))
    for i, (interaction_range, params_vs_N) in zip(color_idx, params_vs_N_vs_range.items()):
        N_list = []
        a_list = []
        b_list = []
        c_list = []
        a_err_list = []
        b_err_list = []
        c_err_list = []
        for N, params in params_vs_N.items():
            N_list.append(N)
            (a, b, c), (a_err, b_err, c_err) = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            a_err_list.append(a_err)
            b_err_list.append(b_err)
            c_err_list.append(c_err)
        plt.errorbar(N_list, b_list, yerr=b_err_list, marker='.', label='b, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Blues')(i))
    plt.legend()
    plt.ylim(bottom=0., top=10.)
    plt.savefig('XXZ_dtwa_bootstrap/plots/b_vs_N_all_range.png')
    plt.ylim(bottom=0.1, top=10.)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_b_vs_log_N_all_range.png')

    fig = plt.figure()
    plt.title('power_law, a + b * |J_eff| ^ c')
    plt.xlabel('N')
    color_idx = np.linspace(1. / len(params_vs_N_vs_range), 1., len(params_vs_N_vs_range))
    for i, (interaction_range, params_vs_N) in zip(color_idx, params_vs_N_vs_range.items()):
        N_list = []
        a_list = []
        b_list = []
        c_list = []
        a_err_list = []
        b_err_list = []
        c_err_list = []
        for N, params in params_vs_N.items():
            N_list.append(N)
            (a, b, c), (a_err, b_err, c_err) = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            a_err_list.append(a_err)
            b_err_list.append(b_err)
            c_err_list.append(c_err)
        plt.errorbar(N_list, c_list, yerr=c_err_list, marker='.', label='c, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Greens')(i))
    plt.legend()
    plt.ylim(bottom=0., top=10.)
    plt.savefig('XXZ_dtwa_bootstrap/plots/c_vs_N_all_range.png')
    plt.ylim(bottom=0.1, top=10.)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_c_vs_log_N_all_range.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, d + e * |J_eff| ^ f')
    plt.xlabel('N')
    color_idx = np.linspace(1. / len(params2_vs_N_vs_range), 1., len(params2_vs_N_vs_range))
    for i, (interaction_range, params_vs_N) in zip(color_idx, params2_vs_N_vs_range.items()):
        N_list = []
        d_list = []
        e_list = []
        f_list = []
        d_err_list = []
        e_err_list = []
        f_err_list = []
        for N, params in params_vs_N.items():
            N_list.append(N)
            (d, e, f), (d_err, e_err, f_err) = params
            d_list.append(d)
            e_list.append(d)
            f_list.append(f)
            d_err_list.append(d_err)
            e_err_list.append(e_err)
            f_err_list.append(f_err)
        plt.errorbar(N_list, d_list, yerr=d_err_list, marker='.', label='d, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/d_vs_N_all_range.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_d_vs_log_N_all_range.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, d + e * |J_eff| ^ f')
    plt.xlabel('N')
    color_idx = np.linspace(1. / len(params2_vs_N_vs_range), 1., len(params2_vs_N_vs_range))
    for i, (interaction_range, params_vs_N) in zip(color_idx, params2_vs_N_vs_range.items()):
        N_list = []
        d_list = []
        e_list = []
        f_list = []
        d_err_list = []
        e_err_list = []
        f_err_list = []
        for N, params in params_vs_N.items():
            N_list.append(N)
            (d, e, f), (d_err, e_err, f_err) = params
            d_list.append(d)
            e_list.append(d)
            f_list.append(f)
            d_err_list.append(d_err)
            e_err_list.append(e_err)
            f_err_list.append(f_err)
        plt.errorbar(N_list, e_list, yerr=e_err_list, marker='.', label='e, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Blues')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/e_vs_N_all_range.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_e_vs_log_N_all_range.png')
    plt.close()

    fig = plt.figure()
    plt.title('power_law, d + e * |J_eff| ^ f')
    plt.xlabel('N')
    color_idx = np.linspace(1. / len(params2_vs_N_vs_range), 1., len(params2_vs_N_vs_range))
    for i, (interaction_range, params_vs_N) in zip(color_idx, params2_vs_N_vs_range.items()):
        N_list = []
        d_list = []
        e_list = []
        f_list = []
        d_err_list = []
        e_err_list = []
        f_err_list = []
        for N, params in params_vs_N.items():
            N_list.append(N)
            (d, e, f), (d_err, e_err, f_err) = params
            d_list.append(d)
            e_list.append(d)
            f_list.append(f)
            d_err_list.append(d_err)
            e_err_list.append(e_err)
            f_err_list.append(f_err)
        plt.errorbar(N_list, f_list, yerr=f_err_list, marker='.', label='f, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Greens')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa_bootstrap/plots/f_vs_N_all_range.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa_bootstrap/plots/log_f_vs_log_N_all_range.png')
    plt.close()

    for N, t_vs_J_eff_vs_range in t_vs_J_eff_vs_range_vs_N.items():
        fig = plt.figure(figsize=(7.2,4.8))
        title = 'N = {}, power law'.format(N)
        plt.title(title)
        plt.xlabel('J_eff')
        plt.ylabel('t_opt')
        color_idx = np.linspace(1. / len(t_vs_J_eff_vs_range), 1., len(t_vs_J_eff_vs_range))
        for i, (interaction_range, t_vs_J_eff) in zip(color_idx, t_vs_J_eff_vs_range.items()):
            J_eff_list = []
            opt_t_list = []
            opt_t_err_list = []
            for J_eff, t in t_vs_J_eff.items():
                opt_t_samples = [t[np.argmin(variance_SN_vs_t)] for variance_SN_vs_t in variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff]]
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    opt_t_list.append(np.mean(opt_t_samples))
                    opt_t_err_list.append(np.std(opt_t_samples, ddof=1))
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(np.mean(opt_t_samples), -0.4, 0.01, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            plt.errorbar(J_eff_list, opt_t_list, yerr=opt_t_err_list, marker='.', linestyle='None', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/opt_t_vs_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

    for N, t_vs_J_eff_vs_range in t_vs_J_eff_vs_range_vs_N.items():
        fig = plt.figure(figsize=(7.2,4.8))
        title = 'N = {}, power law'.format(N)
        plt.title(title)
        plt.xlabel('- J_eff')
        plt.ylabel('t_opt')
        color_idx = np.linspace(1. / len(t_vs_J_eff_vs_range), 1., len(t_vs_J_eff_vs_range))
        for i, (interaction_range, t_vs_J_eff) in zip(color_idx, t_vs_J_eff_vs_range.items()):
            J_eff_list = []
            opt_t_list = []
            opt_t_err_list = []
            for J_eff, t in t_vs_J_eff.items():
                opt_t_samples = [t[np.argmin(variance_SN_vs_t)] for variance_SN_vs_t in variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff]]
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        opt_t_list.append(np.mean(opt_t_samples))
                        opt_t_err_list.append(np.std(opt_t_samples, ddof=1))
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(np.mean(opt_t_samples), 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            plt.errorbar(- np.array(J_eff_list), opt_t_list, yerr=opt_t_err_list, marker='.', linestyle='None', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        plt.plot(- np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), - 1 / np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), label='1 / (- J_eff)', linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/log_opt_t_vs_log_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

    for interaction_range, t_vs_J_eff_vs_N in t_vs_J_eff_vs_N_vs_range.items():
        fig = plt.figure(figsize=(7.2,4.8))
        title = 'exp = {}, power law'.format(interaction_range)
        plt.title(title)
        plt.xlabel('J_eff')
        plt.ylabel('t_opt')
        color_idx = np.linspace(1. / len(t_vs_J_eff_vs_N), 1., len(t_vs_J_eff_vs_N))
        for i, (N, t_vs_J_eff) in zip(color_idx, t_vs_J_eff_vs_N.items()):
            J_eff_list = []
            opt_t_list = []
            opt_t_err_list = []
            for J_eff, t in t_vs_J_eff.items():
                opt_t_samples = [t[np.argmin(variance_SN_vs_t)] for variance_SN_vs_t in variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff]]
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    opt_t_list.append(np.mean(opt_t_samples))
                    opt_t_err_list.append(np.std(opt_t_samples, ddof=1))
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(np.mean(opt_t_samples), -0.4, 0.01, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('N', N, J_eff))
            plt.errorbar(J_eff_list, opt_t_list, yerr=opt_t_err_list, marker='.', linestyle='None', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/opt_t_vs_J_eff_range_{}_all_N.png'.format(interaction_range))
        plt.close()

    for interaction_range, t_vs_J_eff_vs_N in t_vs_J_eff_vs_N_vs_range.items():
        fig = plt.figure(figsize=(7.2,4.8))
        title = 'exp = {}, power law'.format(interaction_range)
        plt.title(title)
        plt.xlabel('- J_eff')
        plt.ylabel('t_opt')
        color_idx = np.linspace(1. / len(t_vs_J_eff_vs_N), 1., len(t_vs_J_eff_vs_N))
        for i, (N, t_vs_J_eff) in zip(color_idx, t_vs_J_eff_vs_N.items()):
            J_eff_list = []
            opt_t_list = []
            opt_t_err_list = []
            for J_eff, t in t_vs_J_eff.items():
                opt_t_samples = [t[np.argmin(variance_SN_vs_t)] for variance_SN_vs_t in variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff]]
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        opt_t_list.append(np.mean(opt_t_samples))
                        opt_t_err_list.append(np.std(opt_t_samples, ddof=1))
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(np.mean(opt_t_samples), 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('N', N, J_eff))
            plt.errorbar(- np.array(J_eff_list), opt_t_list, yerr=opt_t_err_list, marker='.', linestyle='None', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
        plt.plot(- np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), - 1 / np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), label='1 / (- J_eff)', linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa_bootstrap/plots/log_opt_t_vs_log_J_eff_range_{}_all_N.png'.format(interaction_range))
        plt.close()
