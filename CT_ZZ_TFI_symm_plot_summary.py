import numpy as np
import setup
import spin_dynamics as sd
import util
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $\alpha = 0$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')

    N_list = [10,20,50,100,200,500,1000]
    color_idx = np.linspace(1. / len(N_list), 1., len(N_list))
    for i, N in zip(color_idx, N_list):
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        # range_list = [0,0.5,1,1.5,2,2.5,3]
        for interaction_range in [0]:
            J_list = [1.]
            for J in J_list: 
                method_list = ['CT', 'CTv2', 'ZZ', 'XY', 'TFI, h=NJ/2', 'TFI, h=opt', 'TFI, h=0', 'XXZ, J_eff=-0.1', 'XXZ, J_eff=-1.0']
                for method in method_list:
                    if method == 'CT' or method == 'CTv2':
                        dirname = method + '_symm'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                        color = plt.cm.get_cmap('Reds')(i)
                    elif method == 'ZZ' or method == 'XY':
                        dirname = 'ZZ' + '_symm'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, 'power_law', 'exp', interaction_range)
                        color = plt.cm.get_cmap('Blues')(i)
                    elif method == 'TFI, h=NJ/2':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = N * Jz / 2
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                        color = plt.cm.get_cmap('Oranges')(i)
                    elif method == 'TFI, h=opt':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = -2.5 if N <= 20 else -3. if N == 50 else 3.5 if N ==100 else 4. if N == 200 else 5. if N == 500 else 6.
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                        color = plt.cm.get_cmap('Purples')(i)
                    elif method == 'TFI, h=0':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = -0.
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                        color = plt.cm.get_cmap('Greens')(i)
                    elif method == 'XXZ, J_eff=-0.1':
                        method_name = 'XXZ'
                        dirname = method_name + '_symm'
                        J_eff = -0.1
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
                        color = plt.cm.get_cmap('Greys')(i)
                    elif method == 'XXZ, J_eff=-1.0':
                        method_name = 'XXZ'
                        dirname = method_name + '_symm'
                        J_eff = -1.0
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
                        color = plt.cm.get_cmap('YlOrBr')(i)
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                    t = observed_t['t']
                    print(method, len(variance_SN_t), len(t))
                    print('N = {}, t_opt = {}'.format(N, t[np.argmin(variance_SN_t)]))
                    plt.plot(t, variance_SN_t, label=method + ', N = {}'.format(N), color=color)
    plt.ylim(bottom=0., top=1.)
    plt.xlim(left=0., right=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 6})
    # plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/variance_SN_vs_t_power_law_exp_0_all_N.png')

    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $\alpha = 0$')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
    N_list = [10,20,50,100,200,500,1000]
    t_opts = {}
    min_variance_SNs = {}
    for method in ['CT', 'CTv2', 'ZZ', 'XY', 'TFI, h=NJ/2', 'TFI, h=opt', 'TFI, h=0', 'XXZ, J_eff=-0.1', 'XXZ, J_eff=-1.0']:
        min_variance_SN_list = []
        t_opt_list = []
        N_list_ = N_list
        for N in N_list_:
            system_size = (N, 1)
            spin_system = sd.SpinOperators_Symmetry(system_size)
            # range_list = [0,0.5,1,1.5,2,2.5,3]
            for interaction_range in [0]:
                J_list = [1.]
                for J in J_list: 
                    if method == 'CT' or method == 'CTv2':
                        dirname = method + '_symm'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    elif method == 'ZZ' or method == 'XY':
                        dirname = 'ZZ' + '_symm'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, 'power_law', 'exp', interaction_range)
                    elif method == 'TFI, h=NJ/2':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = N * Jz / 2
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                    elif method == 'TFI, h=opt':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = -2.5 if N <= 20 else -3. if N == 50 else 3.5 if N ==100 else 4. if N == 200. else 5. if N == 500 else 6.
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                    elif method == 'TFI, h=0':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = -0.
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                    elif method == 'XXZ, J_eff=-0.1':
                        method_name = 'XXZ'
                        dirname = method_name + '_symm'
                        J_eff = -0.1
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
                    elif method == 'XXZ, J_eff=-1.0':
                        method_name = 'XXZ'
                        dirname = method_name + '_symm'
                        J_eff = -1.0
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)

                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                    t = observed_t['t']
                    # print('N = {}, t_opt = {}'.format(N, t[np.argmin(variance_SN_t)]))
                    min_variance_SN_list.append(min(variance_SN_t[t<=0.5]))
                    t_opt_list.append(t[np.argmin(variance_SN_t[t<=0.5])])
        t_opts[method] = t_opt_list
        min_variance_SNs[method] = min_variance_SN_list
        print(method, N_list_, min_variance_SN_list)
        prev = plt.plot(N_list_, min_variance_SN_list, 'o', label=method)
        prev_color = prev[-1].get_color()
        def fn(N, a, b):
            return a / np.power(N, b)
        popt, pcov = curve_fit(fn, np.array(N_list_)[-3:], min_variance_SN_list[-3:])
        print('ERROR: ', method, pcov)
        plt.plot(N_list_, fn(np.array(N_list_), *popt), label='{} / (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/min_variance_SN_vs_N_power_law_exp_0.png')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/log_min_variance_SN_vs_log_N_power_law_exp_0.png')

    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $\alpha = 0$' + '\n' + r'$- \Delta \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / \Delta t \equiv (1 - \min(N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2)) / t_{opt}$')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$- \Delta \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / \Delta t$')
    N_list = [10,20,50,100,200,500,1000]
    for method in ['CT', 'CTv2', 'ZZ', 'XY', 'TFI, h=NJ/2', 'TFI, h=opt', 'TFI, h=0', 'XXZ, J_eff=-0.1', 'XXZ, J_eff=-1.0']:
        N_list_ = N_list
        min_variance_SN_list = min_variance_SNs[method]
        t_opt_list = t_opts[method]
        init_rate_variance_SN_list = (1 - np.array(min_variance_SN_list)) / np.array(t_opt_list)
        prev = plt.plot(N_list_, init_rate_variance_SN_list, 'o', label=method)
        prev_color = prev[-1].get_color()
        def fn(N, a, b):
            return a * np.power(N, b)
        popt, pcov = curve_fit(fn, np.array(N_list_)[-3:], init_rate_variance_SN_list[-3:])
        plt.plot(N_list_, fn(np.array(N_list_), *popt), label='{} * (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/init_rate_v1_variance_SN_vs_N_power_law_exp_0.png')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/log_init_rate_v1_variance_SN_vs_log_N_power_law_exp_0.png')

    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $\alpha = 0' + '\ninit rate defined as linear part of\nquadratic fit until (1 - opt_squeezing) / (t_opt)')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$- \Delta \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / \Delta t$')
    N_list = [10,20,50,100,200,500,1000]
    for method in ['CT', 'CTv2', 'ZZ', 'XY', 'TFI, h=NJ/2', 'TFI, h=opt', 'TFI, h=0', 'XXZ, J_eff=-0.1', 'XXZ, J_eff=-1.0']:
        N_list_ = N_list
        init_rate_variance_SN_list = []
        for N in N_list_:
            if method == 'CT' or method == 'CTv2':
                dirname = method + '_symm'
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
            elif method == 'ZZ' or method == 'XY':
                dirname = 'ZZ' + '_symm'
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, 'power_law', 'exp', interaction_range)
            elif method == 'TFI, h=NJ/2':
                method_name = 'TFI'
                dirname = method_name + '_symm'
                Jz = -1
                h = N * Jz / 2
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
            elif method == 'TFI, h=opt':
                method_name = 'TFI'
                dirname = method_name + '_symm'
                Jz = -1
                h = -2.5 if N <= 20 else -3. if N == 50 else 3.5 if N ==100 else 4. if N == 200. else 5. if N == 500 else 6.
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
            elif method == 'TFI, h=0':
                method_name = 'TFI'
                dirname = method_name + '_symm'
                Jz = -1
                h = -0.
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
            elif method == 'XXZ, J_eff=-0.1':
                method_name = 'XXZ'
                dirname = method_name + '_symm'
                J_eff = -0.1
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
            elif method == 'XXZ, J_eff=-1.0':
                method_name = 'XXZ'
                dirname = method_name + '_symm'
                J_eff = -1.0
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
            variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
            t = observed_t['t']
            stop_idx = np.argmin(variance_SN_t[t<=0.5]) + 1
            coeffs = np.polyfit(t[:stop_idx], variance_SN_t[:stop_idx], 2) # quadratic, linear, intercept
            init_rate_variance_SN_list.append(- list(coeffs)[1])
        prev = plt.plot(N_list_, init_rate_variance_SN_list, 'o', label=method)
        prev_color = prev[-1].get_color()
        def fn(N, a, b):
            return a * np.power(N, b)
        popt, pcov = curve_fit(fn, np.array(N_list_)[-3:], init_rate_variance_SN_list[-3:])
        plt.plot(N_list_, fn(np.array(N_list_), *popt), label='{} * (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/init_rate_v2_variance_SN_vs_N_power_law_exp_0.png')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/log_init_rate_v2_variance_SN_vs_log_N_power_law_exp_0.png')

    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $\alpha = 0$' + '\ninit rate defined as slope of first time step')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$- \Delta \left( N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2 \right) / \Delta t$')
    N_list = [10,20,50,100,200,500,1000]
    for method in ['CT', 'CTv2', 'ZZ', 'XY', 'TFI, h=NJ/2', 'TFI, h=opt', 'TFI, h=0', 'XXZ, J_eff=-0.1', 'XXZ, J_eff=-1.0']:
        N_list_ = N_list
        init_rate_variance_SN_list = []
        for N in N_list_:
            if method == 'CT' or method == 'CTv2':
                dirname = method + '_symm'
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
            elif method == 'ZZ' or method == 'XY':
                dirname = 'ZZ' + '_symm'
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, 'power_law', 'exp', interaction_range)
            elif method == 'TFI, h=NJ/2':
                method_name = 'TFI'
                dirname = method_name + '_symm'
                Jz = -1
                h = N * Jz / 2
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
            elif method == 'TFI, h=opt':
                method_name = 'TFI'
                dirname = method_name + '_symm'
                Jz = -1
                h = -2.5 if N <= 20 else -3. if N == 50 else 3.5 if N ==100 else 4. if N == 200 else 5. if N == 500 else 6.
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
            elif method == 'TFI, h=0':
                method_name = 'TFI'
                dirname = method_name + '_symm'
                Jz = -1
                h = -0.
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
            elif method == 'XXZ, J_eff=-0.1':
                method_name = 'XXZ'
                dirname = method_name + '_symm'
                J_eff = -0.1
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
            elif method == 'XXZ, J_eff=-1.0':
                method_name = 'XXZ'
                dirname = method_name + '_symm'
                J_eff = -1.0
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
            variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
            t = observed_t['t']
            init_rate_variance_SN_list.append((variance_SN_t[0] - variance_SN_t[1]) / (t[1] - t[0]))
        prev = plt.plot(N_list_, init_rate_variance_SN_list, 'o', label=method)
        prev_color = prev[-1].get_color()
        def fn(N, a, b):
            return a * np.power(N, b)
        popt, pcov = curve_fit(fn, np.array(N_list_)[-3:], init_rate_variance_SN_list[-3:])
        plt.plot(N_list_, fn(np.array(N_list_), *popt), label='{} * (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/init_rate_v0_variance_SN_vs_N_power_law_exp_0.png')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/log_init_rate_v0_variance_SN_vs_log_N_power_law_exp_0.png')

    fig = plt.figure(figsize=(8,6))
    plt.title(r'power law, $\alpha = 0$')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$t_{opt}$')
    N_list = [10,20,50,100,200,500,1000]
    for method in ['CT', 'CTv2', 'ZZ', 'XY', 'TFI, h=NJ/2', 'TFI, h=opt', 'TFI, h=0', 'XXZ, J_eff=-0.1', 'XXZ, J_eff=-1.0']:
        t_opt_list = t_opts[method]
        N_list_ = N_list
        print(method, N_list_, t_opt_list)
        prev = plt.plot(N_list_, t_opt_list, 'o', label=method)
        prev_color = prev[-1].get_color()
        popt, pcov = curve_fit(fn, np.array(N_list_)[-3:], t_opt_list[-3:])
        plt.plot(N_list_, fn(N_list_, *popt), label='{} / (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/t_opt_vs_N_power_law_exp_0.png')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CT_ZZ_TFI_symm/log_t_opt_vs_log_N_power_law_exp_0.png')

    N_list = [10,20,50,100,200,500,1000]
    for i, N in zip(color_idx, N_list):
        system_size = (N, 1)
        spin_system = sd.SpinOperators_Symmetry(system_size)
        fig = plt.figure()
        plt.title(r'$N = {}$, power law, $\alpha = 0$'.format(N))
        plt.xlabel(r'$t$')
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
        # range_list = [0,0.5,1,1.5,2,2.5,3]
        for interaction_range in [0]:
            J_list = [1.]
            for J in J_list: 
                method_list = ['CT', 'CTv2', 'ZZ', 'TFI, h=NJ/2', 'TFI, h=opt', 'TFI, h=0', 'XXZ, J_eff=-0.1', 'XXZ, J_eff=-1.0']
                for method in method_list:
                    if method == 'CT' or method == 'CTv2':
                        dirname = method + '_symm'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    elif method == 'ZZ' or method == 'XY':
                        dirname = 'ZZ' + '_symm'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, 'power_law', 'exp', interaction_range)
                    elif method == 'TFI, h=NJ/2':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = N * Jz / 2
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                    elif method == 'TFI, h=opt':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = -2.5 if N <= 20 else -3. if N == 50 else 3.5 if N ==100 else 4. if N == 200 else 5. if N == 500 else 6.
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                    elif method == 'TFI, h=0':
                        method_name = 'TFI'
                        dirname = method_name + '_symm'
                        Jz = -1
                        h = -0.
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, Jz, h)
                    elif method == 'XXZ, J_eff=-0.1':
                        method_name = 'XXZ'
                        dirname = method_name + '_symm'
                        J_eff = -0.1
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
                    elif method == 'XXZ, J_eff=-1.0':
                        method_name = 'XXZ'
                        dirname = method_name + '_symm'
                        J_eff = -1.0
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_eff_{}'.format(method_name, N, 'power_law', 'exp', interaction_range, J_eff)
                    
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                    t = observed_t['t']
                    prev = plt.plot(t, variance_SN_t, label=method)
                    prev_color = prev[-1].get_color()
                    stop_idx = np.argmin(variance_SN_t[t<=0.5]) + 1
                    coeffs = np.polyfit(t[:stop_idx], variance_SN_t[:stop_idx], 2)
                    p = np.poly1d(coeffs)
                    plt.plot(t[:stop_idx], p(t[:stop_idx]), label='{} + {} t + {} t^2'.format(*reversed([round(c, 2) for c in coeffs])), color=prev_color, linestyle='dashed')
    

        plt.ylim(bottom=0., top=1.)
        plt.xlim(left=0., right=0.5)
        plt.legend()
        plt.savefig('CT_ZZ_TFI_symm/variance_SN_vs_t_N_{}_power_law_exp_0.png'.format(N))



