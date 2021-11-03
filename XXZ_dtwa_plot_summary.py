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
    
    dirname = 'XXZ_dtwa'

    variance_SN_vs_J_eff_vs_range_vs_N = defaultdict(dict)
    t_vs_J_eff_vs_range_vs_N = defaultdict(dict)
    variance_SN_vs_J_eff_vs_N_vs_range = defaultdict(dict)
    t_vs_J_eff_vs_N_vs_range = defaultdict(dict)
    method = 'XXZ'
    min_variance_SN_overall_vs_N_vs_range = defaultdict(list)
    # for N in [10,20,50,100,200]:
    for N in [10,20,50,100]:
        variance_SN_vs_J_eff_vs_range = defaultdict(dict)
        t_vs_J_eff_vs_range = defaultdict(dict)
        # for interaction_range in [0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
        for interaction_range in [0.5,1.0,1.5,2.0,2.5,3.0]:
            variance_SN_vs_J_eff = {}
            t_vs_J_eff = {}
            J_eff_list_full = sorted([1., -1., -0.02, -0.04, -0.06, -0.08, -0.1, -0.12, -0.14, -0.16, -0.18, -0.2, -0.22, -0.24, -0.26, -0.28, -0.3, -0.32, -0.34, -0.36, -0.38, -0.4, 0.01, 0.005, 0.0025, 0.00125, -0.01, -0.005, -0.0025, -0.00125])
            # J_eff_list = sorted(np.concatenate(([1, -1], -1 * np.linspace(0.02, 0.40, 19, endpoint=False), 0.01 * (2 ** (- np.linspace(0, 3, 4))), - 0.01 * (2 ** (- np.linspace(0, 3, 4))))))
            min_variance_SN_overall = 1.
            # fig = plt.figure()
            # plt.title('N = {}, power_law, exp = {}'.format(N, interaction_range))
            # plt.xlabel('t')
            # plt.ylabel('N * <S_a^2> / <S_x>^2')

            for J_eff in J_eff_list_full: 
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method, N, 'power_law', 'exp', interaction_range, J_eff)
                if os.path.isfile('{}/{}'.format(dirname, filename)):
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                    variance_SN_vs_J_eff[J_eff] = variance_SN_t
                    t_vs_J_eff[J_eff] = t
                    min_variance_SN_overall = min(min_variance_SN_overall, min(variance_SN_t))
                    # plt.plot(t, variance_SN_t, label='J_eff = {}'.format(J_eff))
            # plt.ylim(bottom=0., top=1.)
            # plt.legend(prop={'size': 6})
            # plt.savefig('XXZ_dtwa/plots/variance_SN_vs_t_N_{}_power_law_exp_{}_all_J_eff.png'.format(N, interaction_range))
            # plt.close()

            min_variance_SN_overall_vs_N_vs_range[interaction_range].append(min_variance_SN_overall)

            variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range] = variance_SN_vs_J_eff
            t_vs_J_eff_vs_range_vs_N[N][interaction_range] = t_vs_J_eff
            variance_SN_vs_J_eff_vs_N_vs_range[interaction_range][N] = variance_SN_vs_J_eff
            t_vs_J_eff_vs_N_vs_range[interaction_range][N] = t_vs_J_eff


    fig = plt.figure()
    title = 'power_law'
    plt.title(title)
    plt.xlabel('N')
    plt.ylabel('N * <S_a^2> / <S_x>^2')
    for interaction_range, min_variance_SN_overall_vs_N in min_variance_SN_overall_vs_N_vs_range.items():
        plt.plot([10,20,50,100], min_variance_SN_overall_vs_N, label='exp = {}'.format(interaction_range))
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa/plots/min_variance_SN_overall_vs_N_all_ranges.png')
    plt.close()



    def fn1(J_eff, a, b, c):
        return a + b * np.power(np.abs(J_eff), c)

    def fn2(J_eff, a, b, c):
        return a + (b * np.abs(J_eff) / (c + np.abs(J_eff)))

    for N, variance_SN_vs_J_eff_vs_range in variance_SN_vs_J_eff_vs_range_vs_N.items():
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
            for J_eff, variance_SN_t in variance_SN_vs_J_eff.items():
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    min_variance_SN_list.append(min(variance_SN_t))
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(min(variance_SN_t), -0.4, 0.001, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            plt.plot(J_eff_list, min_variance_SN_list, 'o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa/plots/min_variance_SN_vs_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

    params_vs_range_vs_N = defaultdict(dict)
    params_vs_N_vs_range = defaultdict(dict)
    for j, (N, variance_SN_vs_J_eff_vs_range) in enumerate(variance_SN_vs_J_eff_vs_range_vs_N.items()):
        fig = plt.figure(figsize=(7.2,7.2))
        title = 'N = {}, power law'.format(N)
        plt.title(title)
        plt.xlabel('- J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_range), 1., len(variance_SN_vs_J_eff_vs_range))
        for i, (interaction_range, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_range.items()):
            J_eff_list = []
            min_variance_SN_list = []
            for J_eff, variance_SN_t in variance_SN_vs_J_eff.items():
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        min_variance_SN_list.append(min(variance_SN_t))
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(min(variance_SN_t), 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            plt.plot(- np.array(J_eff_list), min_variance_SN_list, 'o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        
            # x = [-J for J in J_eff_list if -0.02 < J < 0]
            # y = [min_variance_SN_list[i] for i in range(len(min_variance_SN_list)) if -0.02 < J_eff_list[i] < 0]
            x = J_eff_list
            y = min_variance_SN_list

            try:
                popt1, pcov1 = curve_fit(fn1, np.array(x), np.array(y))
                params_vs_range_vs_N[N][interaction_range] = tuple(popt1)
                params_vs_N_vs_range[interaction_range][N] = tuple(popt1)
                plt.plot(- np.array(J_eff_list), fn1(np.array(J_eff_list), *popt1), linestyle='dashed', color=plt.cm.get_cmap('Reds')(i), label='%5.3f + %5.3f * |J_eff| ^ %5.3f' % tuple(popt1))
                plt.plot(- np.array(J_eff_list), np.array(min_variance_SN_list) - min_variance_SN_overall_vs_N_vs_range[interaction_range][j], linestyle='dotted', label='shifted, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
            except:
                None


        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa/plots/log_min_variance_SN_vs_log_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

    for interaction_range, variance_SN_vs_J_eff_vs_N in variance_SN_vs_J_eff_vs_N_vs_range.items():
        fig = plt.figure(figsize=(7.2,7.2))
        title = 'exp = {}, power law'.format(interaction_range)
        plt.title(title)
        plt.xlabel('J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        plt.ylim(bottom=0., top=1.)
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_N), 1., len(variance_SN_vs_J_eff_vs_N))

        for i, (N, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_N.items()):
            J_eff_list = []
            min_variance_SN_list = []
            for J_eff, variance_SN_t in variance_SN_vs_J_eff.items():
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    min_variance_SN_list.append(min(variance_SN_t))
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(min(variance_SN_t), -0.4, 0.01, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('N', N, J_eff))
            plt.plot(J_eff_list, min_variance_SN_list, marker='o', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa/plots/min_variance_SN_vs_J_eff_range_{}_all_N.png'.format(interaction_range))
        plt.close()

    for interaction_range, variance_SN_vs_J_eff_vs_N in variance_SN_vs_J_eff_vs_N_vs_range.items():
        fig = plt.figure(figsize=(7.2,7.2))
        title = 'exp = {}, power law'.format(interaction_range)
        plt.title(title)
        plt.xlabel('- J_eff')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_N), 1., len(variance_SN_vs_J_eff_vs_N))
        for i, (N, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_N.items()):
            J_eff_list = []
            min_variance_SN_list = []
            for J_eff, variance_SN_t in variance_SN_vs_J_eff.items():
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        min_variance_SN_list.append(min(variance_SN_t))
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(min(variance_SN_t), 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('N', N, J_eff))
            plt.plot(- np.array(J_eff_list), min_variance_SN_list, 'o', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))

            # x = [-J for J in J_eff_list if -0.02 < J < 0]
            # y = [min_variance_SN_list[i] for i in range(len(min_variance_SN_list)) if -0.02 < J_eff_list[i] < 0]
            x = J_eff_list
            y = min_variance_SN_list

            try:
                popt1, pcov1 = curve_fit(fn1, np.array(x), np.array(y))
                plt.plot(- np.array(J_eff_list), fn1(np.array(J_eff_list), *popt1), linestyle='dashed', color=plt.cm.get_cmap('Reds')(i), label='%5.3f + %5.3f * |J_eff| ^ %5.3f' % tuple(popt1))
            except:
                None
                
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa/plots/log_min_variance_SN_vs_log_J_eff_range_{}_all_N.png'.format(interaction_range))
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
        for interaction_range, params in params_vs_range.items():
            range_list.append(interaction_range)
            a, b, c = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
        plt.plot(range_list, a_list, marker='o', label='a, N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
        # plt.plot(range_list, b_list, label='b, N = {}'.format(N), color=plt.cm.get_cmap('Blues')(i))
        # plt.plot(range_list, c_list, label='c, N = {}'.format(N), color=plt.cm.get_cmap('Greens')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa/plots/a_vs_range_all_N.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa/plots/log_a_vs_log_range_all_N.png')
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
        for interaction_range, params in params_vs_range.items():
            range_list.append(interaction_range)
            a, b, c = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
        # plt.plot(range_list, a_list, label='a, N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
        # plt.plot(range_list, b_list, label='b, N = {}'.format(N), color=plt.cm.get_cmap('Blues')(i))
        plt.plot(range_list, c_list, marker='o', label='c, N = {}'.format(N), color=plt.cm.get_cmap('Greens')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa/plots/c_vs_range_all_N.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa/plots/log_c_vs_log_range_all_N.png')
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
        for N, params in params_vs_N.items():
            N_list.append(N)
            a, b, c = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
        plt.plot(N_list, a_list, marker='o', label='a, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        # plt.plot(N_list, b_list, label='b, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Blues')(i))
        # plt.plot(N_list, c_list, label='c, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Greens')(i))
    plt.plot(N_list, 1 / np.array(N_list), linestyle='dashed', label='1 / N')
    plt.plot(N_list, 1 / np.power(N_list, 2/3), linestyle='dashed', label='1 / (N ^ (2/3))')
    plt.legend()
    plt.savefig('XXZ_dtwa/plots/a_vs_N_all_range.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa/plots/log_a_vs_log_N_all_range.png')
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
        for N, params in params_vs_N.items():
            N_list.append(N)
            a, b, c = params
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
        # plt.plot(N_list, a_list, label='a, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        # plt.plot(N_list, b_list, label='b, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Blues')(i))
        plt.plot(N_list, c_list, marker='o', label='c, exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Greens')(i))
    plt.legend()
    plt.savefig('XXZ_dtwa/plots/c_vs_N_all_range.png')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_dtwa/plots/log_c_vs_log_N_all_range.png')
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
            for J_eff, t in t_vs_J_eff.items():
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    opt_t_list.append(t[np.argmin(variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff])])
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(t[np.argmin(variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff])], -0.4, 0.01, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            plt.plot(J_eff_list, opt_t_list, 'o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa/plots/opt_t_vs_J_eff_N_{}_all_ranges.png'.format(N))
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
            for J_eff, t in t_vs_J_eff.items():
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        opt_t_list.append(t[np.argmin(variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff])])
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(t[np.argmin(variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff])], 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('exp', interaction_range, J_eff))
            plt.plot(- np.array(J_eff_list), opt_t_list, 'o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        plt.plot(- np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), - 1 / np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), label='1 / (- J_eff)', linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa/plots/log_opt_t_vs_log_J_eff_N_{}_all_ranges.png'.format(N))
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
            for J_eff, t in t_vs_J_eff.items():
                if J_eff not in [1., -1.]:
                    J_eff_list.append(J_eff)
                    opt_t_list.append(t[np.argmin(variance_SN_vs_J_eff_vs_N_vs_range[interaction_range][N][J_eff])])
                else:
                    color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                    linestyle = 'dashed' if J_eff == 1. else 'solid'
                    plt.hlines(t[np.argmin(variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff])], -0.4, 0.01, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('N', N, J_eff))
            plt.plot(J_eff_list, opt_t_list, 'o', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.tight_layout()
        plt.savefig('XXZ_dtwa/plots/opt_t_vs_J_eff_range_{}_all_N.png'.format(interaction_range))
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
            for J_eff, t in t_vs_J_eff.items():
                if J_eff < 0:
                    if J_eff not in [1., -1.]:
                        J_eff_list.append(J_eff)
                        opt_t_list.append(t[np.argmin(variance_SN_vs_J_eff_vs_N_vs_range[interaction_range][N][J_eff])])
                    else:
                        color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                        linestyle = 'dashed' if J_eff == 1. else 'solid'
                        plt.hlines(t[np.argmin(variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff])], 0.001, 0.4, color=color, linestyle='dashed', label='{} = {}, J_eff = {}'.format('N', N, J_eff))
            plt.plot(- np.array(J_eff_list), opt_t_list, 'o', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))
        plt.plot(- np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), - 1 / np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), label='1 / (- J_eff)', linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_dtwa/plots/log_opt_t_vs_log_J_eff_range_{}_all_N.png'.format(interaction_range))
        plt.close()
