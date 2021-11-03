import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import setup
import spin_dynamics as sd
import util
from collections import defaultdict
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    dirname = 'XXZ_symm'

    variance_SN_vs_J_eff_vs_range_vs_N = defaultdict(dict)
    t_vs_J_eff_vs_range_vs_N = defaultdict(dict)
    variance_SN_vs_J_eff_vs_N_vs_range = defaultdict(dict)
    t_vs_J_eff_vs_N_vs_range = defaultdict(dict)
    method = 'XXZ'
    min_variance_SN_overall_vs_N_vs_range = defaultdict(list)
    min_variance_SN_vs_N_vs_range = defaultdict(dict)
    init_rate_variance_SN_overall_vs_N_vs_range = defaultdict(list)
    init_rate_variance_SN_vs_N_vs_range = defaultdict(dict)
    for N in [10,20,50,100,200,500,1000]:
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        variance_SN_vs_J_eff_vs_range = defaultdict(dict)
        t_vs_J_eff_vs_range = defaultdict(dict)
        for interaction_range in [0]:
            variance_SN_vs_J_eff = {}
            t_vs_J_eff = {}
            J_eff_list_full = sorted([-0.02, -0.04, -0.06, -0.08, -0.1, 0.01, 0.005, 0.0025, 0.00125, -0.01, -0.005, -0.0025, -0.00125])
            # J_eff_list_full = sorted([1., -1., -0.02, -0.04, -0.06, -0.08, -0.1, -0.12, -0.14, -0.16, -0.18, -0.2, -0.22, -0.24, -0.26, -0.28, -0.3, -0.32, -0.34, -0.36, -0.38, -0.4, 0.01, 0.005, 0.0025, 0.00125, -0.01, -0.005, -0.0025, -0.00125])
            # J_eff_list = sorted(np.concatenate(([1, -1], -1 * np.linspace(0.02, 0.40, 19, endpoint=False), 0.01 * (2 ** (- np.linspace(0, 3, 4))), - 0.01 * (2 ** (- np.linspace(0, 3, 4))))))
            min_variance_SN_overall = 1.
            init_rate_variance_SN_overall = 0
            for J_eff in J_eff_list_full: 
                if J_eff not in min_variance_SN_vs_N_vs_range[interaction_range]:
                    min_variance_SN_vs_N_vs_range[interaction_range][J_eff] = []
                    init_rate_variance_SN_vs_N_vs_range[interaction_range][J_eff] = []
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method, N, 'power_law', 'exp', interaction_range, J_eff)
                if os.path.isfile('{}/{}'.format(dirname, filename)):
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                    t = observed_t['t']
                    variance_SN_t = variance_SN_t[t < 15]
                    variance_norm_t = variance_norm_t[t < 15]
                    angle_t = angle_t[t < 15]
                    t = t[t < 15]

                    variance_SN_vs_J_eff[J_eff] = variance_SN_t
                    t_vs_J_eff[J_eff] = t
                    min_variance_SN_overall = min(min_variance_SN_overall, min(variance_SN_t))
                    min_variance_SN_vs_N_vs_range[interaction_range][J_eff].append(min(variance_SN_t))
                    init_rate_variance_SN = (variance_SN_t[1]-variance_SN_t[0])/(t[1]-t[0])
                    init_rate_variance_SN_overall = min(init_rate_variance_SN_overall, init_rate_variance_SN)
                    init_rate_variance_SN_vs_N_vs_range[interaction_range][J_eff].append(init_rate_variance_SN)
                    

            print(N, min_variance_SN_overall)
            min_variance_SN_overall_vs_N_vs_range[interaction_range].append(min_variance_SN_overall)
            init_rate_variance_SN_overall_vs_N_vs_range[interaction_range].append(init_rate_variance_SN_overall)

            variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range] = variance_SN_vs_J_eff
            t_vs_J_eff_vs_range_vs_N[N][interaction_range] = t_vs_J_eff
            variance_SN_vs_J_eff_vs_N_vs_range[interaction_range][N] = variance_SN_vs_J_eff
            t_vs_J_eff_vs_N_vs_range[interaction_range][N] = t_vs_J_eff

    Fig = plt.figure()
    plt.title('power_law, exp = 0, J_eff = {}'.format(-0.1))
    plt.ylabel('N * <S_a^2> / <S_x>^2')
    plt.xlabel('t')
    for N in [10,20,50,100,200,500,1000]:
        plt.plot(t_vs_J_eff_vs_range_vs_N[N][0][-0.1], variance_SN_vs_J_eff_vs_range_vs_N[N][0][-0.1], label='N = {}'.format(N))
    plt.legend()
    plt.ylim(bottom=0., top=1.)
    plt.savefig('XXZ_symm/plots/variance_SN_vs_t_power_law_exp_0_all_N.png')

    fig = plt.figure(figsize=(10,10))
    title = 'power_law'
    plt.title(title)
    plt.xlabel('N')
    plt.ylabel('N * <S_a^2> / <S_x>^2')
    def fn0(x, a, b):
        return a * np.power(x, b)
    for interaction_range, min_variance_SN_overall_vs_N in min_variance_SN_overall_vs_N_vs_range.items():
        plt.plot([10,20,50,100,200,500,1000], min_variance_SN_overall_vs_N, 'o', label='exp = {}, min over J_eff'.format(interaction_range))
        popt0, pcov0 = curve_fit(fn0, np.array([50,100,200,500,1000]), min_variance_SN_overall_vs_N[2:])
        plt.plot([10,20,50,100,200,500,1000], fn0(np.array([10,20,50,100,200,500,1000]), *popt0), linestyle='dashed', label='%5.3f * N ^ %5.3f' % tuple(popt0))
        for J_eff, min_variance_SN_vs_N in min_variance_SN_vs_N_vs_range[interaction_range].items():
            plt.plot([10,20,50,100,200,500,1000], min_variance_SN_vs_N, 'o', label='exp = {}, J_eff = {}'.format(interaction_range, J_eff))
    plt.plot([10,20,50,100,200,500,1000], 1. / np.array([10,20,50,100,200,500,1000]), linestyle='dashed', label='1 / N')
    plt.plot([10,20,50,100,200,500,1000], 1. / np.power([10,20,50,100,200,500,1000], 2/4), linestyle='dashed', label='1 / (N ^ (2/3))')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_symm/plots/min_variance_SN_overall_vs_N_all_ranges.png')
    plt.close()

    fig = plt.figure(figsize=(10,10))
    title = 'power law'
    plt.title(title)
    plt.xlabel('N')
    plt.ylabel('- Δ (N * <S_a^2> / <S_x>^2) / Δt')
    def fn0(x, a, b):
        return a * np.power(x, b)
    for interaction_range, init_rate_variance_SN_overall_vs_N in init_rate_variance_SN_overall_vs_N_vs_range.items():
        prev = plt.plot([10,20,50,100,200,500,1000], -np.array(init_rate_variance_SN_overall_vs_N), 's', label='exp = {}, min over J_eff'.format(interaction_range))
        popt0, pcov0 = curve_fit(fn0, np.array([10,20,50,100,200,500,1000]), -np.array(init_rate_variance_SN_overall_vs_N))
        plt.plot([10,20,50,100,200,500,1000], fn0(np.array([10,20,50,100,200,500,1000]), *popt0), linestyle='dashed', color=prev[-1].get_color(), label='%5.3f * N ^ %5.3f' % tuple(popt0))
        init_rate_param_vs_J_eff = {}
        for J_eff, init_rate_variance_SN_vs_N in init_rate_variance_SN_vs_N_vs_range[interaction_range].items():
            prev = plt.plot([10,20,50,100,200,500,1000], -np.array(init_rate_variance_SN_vs_N), 'o', label='exp = {}, J_eff = {}'.format(interaction_range, J_eff))
            popt0, pcov0 = curve_fit(fn0, np.array([10,20,50,100,200,500,1000]), -np.array(init_rate_variance_SN_vs_N))
            plt.plot([10,20,50,100,200,500,1000], fn0(np.array([10,20,50,100,200,500,1000]), *popt0), linestyle='dashed', color=prev[-1].get_color(), label='%5.3f * N ^ %5.3f' % tuple(popt0))
            init_rate_param_vs_J_eff[J_eff] = tuple(popt0)[0]
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('XXZ_symm/plots/init_rate_variance_SN_overall_vs_N_all_ranges.png')
    plt.close()

    J_eff_list = []
    param_list = []
    for J_eff, param in init_rate_param_vs_J_eff.items():
        if J_eff < 0:
            J_eff_list.append(- J_eff)
            param_list.append(param)
    fig = plt.figure()
    plt.title('power law, exp = 0, - Δ (N * <S_a^2> / <S_x>^2) / Δt = a * N ^ b')
    plt.xlabel('- J_eff')
    plt.ylabel('a')
    plt.plot(J_eff_list, param_list, 'o')
    slope, intercept, r, p, se = linregress(J_eff_list, param_list)
    plt.plot(J_eff_list, slope * np.array(J_eff_list) + intercept, linestyle='dashed', color=prev[-1].get_color(), label='%5.3f * J_eff + %5.3f' % (slope, intercept))
    plt.legend()
    plt.savefig('XXZ_symm/plots/init_rate_params_vs_J_eff.png')



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
                print(N, J_eff, min(variance_SN_t))
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
        plt.savefig('XXZ_symm/plots/min_variance_SN_vs_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

    for N, variance_SN_vs_J_eff_vs_range in variance_SN_vs_J_eff_vs_range_vs_N.items():
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
            plt.plot(- np.array(J_eff_list), min_variance_SN_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_symm/plots/log_min_variance_SN_vs_log_J_eff_N_{}_all_ranges.png'.format(N))
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
            print('N = {}, J_eff = {}, min_variance_SN = {}'.format(N, J_eff_list, min_variance_SN_list))
            plt.plot(J_eff_list, min_variance_SN_list, marker='o', label='N = {}'.format(N), color=plt.cm.get_cmap('Reds')(i))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.tight_layout()
        plt.savefig('XXZ_symm/plots/min_variance_SN_vs_J_eff_range_{}_all_N.png'.format(interaction_range))
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
                
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 6})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_symm/plots/log_min_variance_SN_vs_log_J_eff_range_{}_all_N.png'.format(interaction_range))
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
            print('N = {}, J_eff = {}, t_opt = {}'.format(N, J_eff_list, opt_t_list))
            plt.plot(J_eff_list, opt_t_list, 'o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.tight_layout()
        plt.savefig('XXZ_symm/plots/opt_t_vs_J_eff_N_{}_all_ranges.png'.format(N))
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
            popt0, pcov0 = curve_fit(fn0, - np.array(J_eff_list)[- np.array(J_eff_list) > 0.01], np.array(opt_t_list)[- np.array(J_eff_list) > 0.01])
            plt.plot(- np.array(J_eff_list), fn0(- np.array(J_eff_list), *popt0), linestyle='dashed', label='%5.3f * |J_eff| ^ %5.3f' % tuple(popt0))
        plt.plot(- np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), - 0.2 / np.array([J for J in J_eff_list_full if J < 0 and J != 1.0 and J != -1.0]), label='1 / (- J_eff)', linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('XXZ_symm/plots/log_opt_t_vs_log_J_eff_N_{}_all_ranges.png'.format(N))
        plt.close()

    fig = plt.figure(figsize=(7.2,4.8))
    title = 'power_law, exp = 0'
    plt.title(title)
    plt.xlabel('N')
    plt.ylabel('- J_eff * t_opt')
    N_list = []
    Jefft_mean_list = []
    Jefft_std_list = []

    for N, t_vs_J_eff_vs_range in t_vs_J_eff_vs_range_vs_N.items():
        color_idx = np.linspace(1. / len(t_vs_J_eff_vs_range), 1., len(t_vs_J_eff_vs_range))
        Jefft_list = []
        Jefft_list2 = []
        for i, (interaction_range, t_vs_J_eff) in zip(color_idx, t_vs_J_eff_vs_range.items()):
            J_eff_list = []
            opt_t_list = []
            for J_eff, t in t_vs_J_eff.items():
                if J_eff < -0.01:
                    Jefft_list.append(-J_eff * t[np.argmin(variance_SN_vs_J_eff_vs_range_vs_N[N][interaction_range][J_eff])])
        print(len(Jefft_list))
        N_list.append(N)
        Jefft_mean_list.append(np.mean(Jefft_list))
        Jefft_std_list.append(np.std(Jefft_list))
    plt.errorbar(N_list, Jefft_mean_list, yerr=Jefft_std_list, marker='o', capsize=4, label='J_eff < -0.01')
    popt0, pcov0 = curve_fit(fn0, np.array(N_list), np.array(Jefft_mean_list))
    plt.plot(N_list, fn0(np.array(N_list), *popt0), linestyle='dashed', label='%5.3f * N ^ %5.3f' % tuple(popt0))
    plt.plot(N_list, 0.8 * np.power(N_list, -0.5), linestyle='dashed', label='%5.3f * N ^ %5.3f' % (0.8, -0.5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('XXZ_symm/plots/log_opt_J_efft_vs_log_N_all_ranges.png')
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
        plt.savefig('XXZ_symm/plots/opt_t_vs_J_eff_range_{}_all_N.png'.format(interaction_range))
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
        plt.savefig('XXZ_symm/plots/log_opt_t_vs_log_J_eff_range_{}_all_N.png'.format(interaction_range))
        plt.close()
