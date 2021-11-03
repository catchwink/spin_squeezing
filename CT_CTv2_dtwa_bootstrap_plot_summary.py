import numpy as np
import setup
import spin_dynamics as sd
import util
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    def fn(N, a, b):
        return a / np.power(N, b)

    range_list = [0,0.5,1,1.5,2,2.5,3]
    for interaction_range in range_list:
        fig = plt.figure(figsize=(8,6))
        plt.title('power law, exp = {}'.format(interaction_range))
        plt.xlabel('t')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        N_list = [10,20,50,100,200]
        # N_list = [200]
        color_idx = np.linspace(1. / len(N_list), 1., len(N_list))
        for i, N in zip(color_idx, N_list):
            J_list = [1.]
            for J in J_list: 
                for method in ['CT', 'CTv2']:
                    variance_SN_t_list = []
                    variance_norm_t_list = []
                    angle_t_list = []
                    t_list = []
                    for bs in range(50):
                        dirname = method + '_dtwa_bootstrap'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J, bs)
                        if method == 'CT':
                            color = plt.cm.get_cmap('Reds')(i)
                            linestyle = 'solid'
                        else:
                            color = plt.cm.get_cmap('Blues')(i)
                            linestyle = 'dashed'
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                        variance_SN_t_list.append(variance_SN_t)
                        variance_norm_t_list.append(variance_norm_t)
                        angle_t_list.append(angle_t)
                        t_list.append(t)
                    variance_SN_t_mean = np.mean(variance_SN_t_list, axis=0)
                    variance_norm_t_mean = np.mean(variance_norm_t_list, axis=0)
                    angle_t_mean = np.mean(angle_t_list, axis=0)
                    t = t_list[0]
                    variance_SN_t_std = np.std(variance_SN_t_list, axis=0, ddof=1)
                    variance_norm_t_std = np.std(variance_norm_t_list, axis=0, ddof=1)
                    angle_t_std = np.std(angle_t_list, axis=0, ddof=1)
                    print('N = {}, t_opt = {}'.format(N, t[np.argmin(variance_SN_t_mean)]))
                    plt.errorbar(t, variance_SN_t_mean, yerr=variance_SN_t_std, label=method + ', N = {}, '.format(N), color=color, linestyle=linestyle)
        plt.ylim(bottom=0., top=1.)
        plt.xlim(left=0., right=1.2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
        plt.tight_layout()
        plt.savefig('CTv2_dtwa_bootstrap/plots/variance_SN_vs_t_power_law_exp_{}_all_N.png'.format(interaction_range))
        plt.close()

    fig = plt.figure(figsize=(8,6))
    plt.title('power law')
    plt.xlabel('N')
    plt.ylabel('N * <S_a^2> / <S_x>^2')
    t_opts_vs_range = {}
    color_idx = np.linspace(1. / len(range_list), 1., len(range_list))
    for i, interaction_range in zip(color_idx, range_list):
        print('RANGE = ', interaction_range)
        N_list = [10,20,50,100,200]
        t_opts = {}
        for method in ['CT', 'CTv2']:
            min_variance_SN_list = []
            t_opt_list = []
            min_variance_SN_err_list = []
            t_opt_err_list = []
            for N in N_list:
            
                J_list = [1.]
                for J in J_list: 
                    min_variance_SN_t_sample_list = []
                    t_opt_sample_list = []
                    for bs in range(50):
                        dirname = method + '_dtwa_bootstrap'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J, bs)
                        if method == 'CT':
                            color = plt.cm.get_cmap('Reds')(i)
                            linestyle = 'solid'
                        else:
                            color = plt.cm.get_cmap('Blues')(i)
                            linestyle = 'dashed'

                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                        min_variance_SN_t_sample_list.append(min(variance_SN_t))
                        t_opt_sample_list.append(t[np.argmin(variance_SN_t)])
                    min_variance_SN_list.append(np.mean(min_variance_SN_t_sample_list))
                    t_opt_list.append(np.mean(t_opt_sample_list))
                    min_variance_SN_err_list.append(np.std(min_variance_SN_t_sample_list, ddof=1))
                    t_opt_err_list.append(np.std(t_opt_sample_list, ddof=1))
            t_opts[method] = (t_opt_list, t_opt_err_list)
            c = plt.get_cmap('Reds')(i) if method == 'CT' else plt.get_cmap('Blues')(i)
            print(min_variance_SN_err_list)
            prev = plt.errorbar(N_list, min_variance_SN_list, yerr=min_variance_SN_err_list, marker='o', linestyle='None', label='{}, exp = {}'.format(method, interaction_range), color=c)
            prev_color = prev.lines[0].get_color()
            popt, pcov = curve_fit(fn, np.array(N_list)[2:], min_variance_SN_list[2:])
            plt.plot(N_list, fn(N_list, *popt), label='{} / (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
        t_opts_vs_range[interaction_range] = t_opts
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.ylim(bottom=0., top=1.)
    plt.savefig('CTv2_dtwa_bootstrap/plots/min_variance_SN_vs_N_power_law_all_ranges.png')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=None, top=None)
    plt.tight_layout()
    plt.savefig('CTv2_dtwa_bootstrap/plots/log_min_variance_SN_vs_log_N_power_law_all_ranges.png')

    fig = plt.figure(figsize=(8,6))
    plt.title('N = 200, power law')
    plt.xlabel('exp')
    plt.ylabel('N * <S_a^2> / <S_x>^2')
    color_idx = np.linspace(1. / len(range_list), 1., len(range_list))
    for method in ['CT', 'CTv2']:
        min_variance_SN_list = []
        min_variance_SN_err_list = []
        for i, interaction_range in zip(color_idx, range_list):
        
            N = 200
            
            J_list = [1.]
            for J in J_list: 
                min_variance_SN_t_sample_list = []
                for bs in range(50):
                    dirname = method + '_dtwa_bootstrap'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J, bs)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'

                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                    min_variance_SN_t_sample_list.append(min(variance_SN_t))
                min_variance_SN_list.append(np.mean(min_variance_SN_t_sample_list))
                min_variance_SN_err_list.append(np.std(min_variance_SN_t_sample_list, ddof=1))

        plt.errorbar(range_list, min_variance_SN_list, yerr=min_variance_SN_err_list, marker='o', linestyle='None', color=color, label=method)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.ylim(bottom=0., top=1.)
    plt.savefig('CTv2_dtwa_bootstrap/plots/min_variance_SN_vs_range_power_law_N_200.png'.format(interaction_range))
    plt.ylim(bottom=1./N, top=1.)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CTv2_dtwa_bootstrap/plots/log_min_variance_SN_vs_range_power_law_N_200.png'.format(interaction_range))

    fig = plt.figure(figsize=(8,6))
    plt.title('N = 200, power law')
    plt.xlabel('exp')
    plt.ylabel('<S_a^2> (normalized)')
    color_idx = np.linspace(1. / len(range_list), 1., len(range_list))

    for method in ['CT', 'CTv2']:
        min_variance_norm_list = []
        min_variance_norm_err_list = []
        for i, interaction_range in zip(color_idx, range_list):
        
            N = 200
            
            J_list = [1.]
            for J in J_list: 
                min_variance_norm_t_sample_list = []
                for bs in range(50):
                    dirname = method + '_dtwa_bootstrap'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J, bs)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'

                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                    min_variance_norm_t_sample_list.append(variance_norm_t[np.argmin(variance_SN_t)])
                min_variance_norm_list.append(np.mean(min_variance_norm_t_sample_list))
                min_variance_norm_err_list.append(np.std(min_variance_norm_t_sample_list, ddof=1))

        plt.errorbar(range_list, min_variance_norm_list, yerr=min_variance_norm_err_list, marker='o', linestyle='None', color=color, label=method)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.ylim(bottom=0., top=1.)
    plt.savefig('CTv2_dtwa_bootstrap/plots/min_variance_norm_vs_range_power_law_N_200.png'.format(interaction_range))
    plt.ylim(bottom=1./N, top=1.)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CTv2_dtwa_bootstrap/plots/log_min_variance_norm_vs_range_power_law_N_200.png'.format(interaction_range))

    fig = plt.figure(figsize=(8,6))
    plt.title('N = 200, power law')
    plt.xlabel('exp')
    plt.ylabel('<S_x>^2 (normalized)')
    color_idx = np.linspace(1. / len(range_list), 1., len(range_list))

    for method in ['CT', 'CTv2']:
        signal_list = []
        signal_err_list = []
        for i, interaction_range in zip(color_idx, range_list):
        
            N = 200
            
            J_list = [1.]
            for J in J_list: 
                signal_t_sample_list = []
                for bs in range(50):
                    dirname = method + '_dtwa_bootstrap'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J, bs)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'

                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, Sx, t = observed_t['min_variance_SN'], observed_t['S_x'], observed_t['t']
                    signal_t_sample_list.append(np.power(Sx[np.argmin(variance_SN_t)] / Sx[0], 2))
                signal_list.append(np.mean(signal_t_sample_list))
                signal_err_list.append(np.std(signal_t_sample_list, ddof=1))

        plt.errorbar(range_list, signal_list, yerr=signal_err_list, marker='o', linestyle='None', color=color, label=method)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    # plt.ylim(bottom=0., top=1.)
    plt.savefig('CTv2_dtwa_bootstrap/plots/signal_vs_range_power_law_N_200.png'.format(interaction_range))
    # plt.ylim(bottom=1./N, top=1.)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CTv2_dtwa_bootstrap/plots/log_signal_vs_log_range_power_law_N_200.png'.format(interaction_range))

    
    for interaction_range in [0,0.5,1,1.5,2,2.5,3]:
        t_opts = t_opts_vs_range[interaction_range]
        fig = plt.figure(figsize=(8,6))
        plt.title('power law, exp = {}'.format(interaction_range))
        plt.xlabel('N')
        plt.ylabel('t_opt')
        N_list = [10,20,50,100,200]
        for method in ['CT', 'CTv2']:
            t_opt_list, t_opt_err_list = t_opts[method]
            prev = plt.errorbar(N_list, t_opt_list, yerr=t_opt_err_list, marker='o', linestyle='None', label=method)
            prev_color = prev.lines[0].get_color()
            popt, pcov = curve_fit(fn, np.array(N_list)[2:], t_opt_list[2:])
            plt.plot(N_list, fn(N_list, *popt), label='{} / (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
        plt.tight_layout()
        plt.savefig('CTv2_dtwa_bootstrap/plots/t_opt_vs_N_power_law_exp_{}.png'.format(interaction_range))
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('CTv2_dtwa_bootstrap/plots/log_t_opt_vs_log_N_power_law_exp_{}.png'.format(interaction_range))

    
    N_list = [10,20,50,100,200]
    for i, N in zip(color_idx, N_list):
        fig = plt.figure()
        plt.title('N = {}, power law'.format(N))
        plt.xlabel('t')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        for interaction_range in range_list:
            J_list = [1.]
            for J in J_list: 
                for method in ['CT', 'CTv2']:
                    variance_SN_t_list = []
                    for bs in range(50):
                        dirname = method + '_dtwa_bootstrap'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J, bs)
                        if method == 'CT':
                            color = plt.cm.get_cmap('Reds')(i)
                            linestyle = 'solid'
                        else:
                            color = plt.cm.get_cmap('Blues')(i)
                            linestyle = 'dashed'
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                        variance_SN_t_list.append(variance_SN_t)
                    variance_SN_t_mean = np.mean(variance_SN_t_list, axis=0)
                    variance_SN_t_std = np.std(variance_SN_t_list, axis=0, ddof=1)
                    plt.errorbar(t, variance_SN_t_mean, yerr=variance_SN_t_std, label='{}, exp = {}'.format(method, interaction_range))
        plt.ylim(bottom=0., top=1.)
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('CTv2_dtwa_bootstrap/plots/variance_SN_vs_t_N_{}_power_law_all_ranges.png'.format(N))
        plt.yscale('log')
        plt.ylim(bottom=1./N, top=1.)
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('CTv2_dtwa_bootstrap/plots/log_variance_SN_vs_t_N_{}_power_law_all_ranges.png'.format(N))

    N_list = [200]
    for i, N in zip(color_idx, N_list):
        fig = plt.figure()
        plt.title('N = {}, power law'.format(N))
        plt.xlabel('t')
        plt.ylabel('<S_a^2> (normalized)')
        for interaction_range in [0, 1, 3]:
            J_list = [1.]
            for J in J_list: 
                for method in ['CT', 'CTv2']:
                    variance_norm_t_list = []
                    for bs in range(50):
                        dirname = method + '_dtwa_bootstrap'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J, bs)
                        if method == 'CT':
                            color = plt.cm.get_cmap('Reds')(i)
                            linestyle = 'solid'
                        else:
                            color = plt.cm.get_cmap('Blues')(i)
                            linestyle = 'dashed'
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_norm_t, Sx, t = observed_t['min_variance_norm'], observed_t['S_x'], observed_t['t']
                        variance_norm_t_list.append(variance_norm_t)
                    variance_norm_t_mean = np.mean(variance_norm_t_list, axis=0)
                    variance_norm_t_std = np.std(variance_norm_t_list, axis=0, ddof=1)
                    plt.errorbar(t, variance_norm_t_mean, yerr=variance_norm_t_std, label='{}, N = {}, exp = {}'.format(method, N, interaction_range), linestyle=linestyle)
        plt.ylim(bottom=1./N, top=1.)
        plt.yscale('log')
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('CTv2_dtwa_bootstrap/plots/log_variance_norm_vs_t_N_{}_power_law_all_ranges.png'.format(N))

    N_list = [200]
    for i, N in zip(color_idx, N_list):
        fig = plt.figure()
        plt.title('N = {}, power law'.format(N))
        plt.xlabel('t')
        plt.ylabel('<S_x>^2 (normalized)')
        for interaction_range in [0,1,3]:
            J_list = [1.]
            for J in J_list: 
                for method in ['CT', 'CTv2']:
                    signal_t_list = []
                    for bs in range(50):
                        dirname = method + '_dtwa_bootstrap'
                        filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}/bootstrap_{}'.format(method, N, 'power_law', 'exp', interaction_range, J, bs)
                        if method == 'CT':
                            color = plt.cm.get_cmap('Reds')(i)
                            linestyle = 'solid'
                        else:
                            color = plt.cm.get_cmap('Blues')(i)
                            linestyle = 'dashed'
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_norm_t, Sx, t = observed_t['min_variance_norm'], observed_t['S_x'], observed_t['t']
                        signal_t_list.append(np.power(Sx / Sx[0], 2))
                    signal_t_mean = np.mean(signal_t_list, axis=0)
                    signal_t_std = np.std(signal_t_list, axis=0, ddof=1)
                    plt.errorbar(t, signal_t_mean, yerr=signal_t_std, label='{}, N = {}, exp = {}'.format(method, N, interaction_range), linestyle=linestyle)
        plt.ylim(bottom=1./N, top=1.)
        plt.yscale('log')
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('CTv2_dtwa_bootstrap/plots/log_signal_vs_t_N_{}_power_law_all_ranges.png'.format(N))



