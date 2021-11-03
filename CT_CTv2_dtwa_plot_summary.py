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
                    dirname = method + '_dtwa'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                    print(variance_SN_t)
                    print('N = {}, t_opt = {}'.format(N, t[np.argmin(variance_SN_t)]))
                    plt.plot(t, variance_SN_t, label=method + ', N = {}, '.format(N), color=color, linestyle=linestyle)
        plt.ylim(bottom=0., top=1.)
        plt.xlim(left=0., right=1.2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
        plt.tight_layout()
        plt.savefig('CTv2_dtwa/plots/variance_SN_vs_t_power_law_exp_{}_all_N.png'.format(interaction_range))
        plt.close()

    fig = plt.figure(figsize=(8,6))
    plt.title('power law, exp = {}'.format(interaction_range))
    plt.xlabel('N')
    plt.ylabel('N * <S_a^2> / <S_x>^2')
    t_opts_vs_range = {}
    color_idx = np.linspace(1. / len(range_list), 1., len(range_list))
    for i, interaction_range in zip(color_idx, range_list):
        
        N_list = [10,20,50,100,200]
        t_opts = {}
        for method in ['CT', 'CTv2']:
            min_variance_SN_list = []
            t_opt_list = []
            for N in N_list:
            
                J_list = [1.]
                for J in J_list: 
                    
                    dirname = method + '_dtwa'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'

                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                    print('N = {}, t_opt = {}'.format(N, t[np.argmin(variance_SN_t)]))
                    min_variance_SN_list.append(min(variance_SN_t))
                    t_opt_list.append(t[np.argmin(variance_SN_t)])
            t_opts[method] = t_opt_list
            c = plt.get_cmap('Reds')(i) if method == 'CT' else plt.get_cmap('Blues')(i)
            prev = plt.plot(N_list, min_variance_SN_list, 'o', label='{}, exp = {}'.format(method, interaction_range), color=c)
            prev_color = prev[-1].get_color()
            popt, pcov = curve_fit(fn, np.array(N_list)[2:], min_variance_SN_list[2:])
            plt.plot(N_list, fn(N_list, *popt), label='{} / (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
        t_opts_vs_range[interaction_range] = t_opts
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.ylim(bottom=0., top=1.)
    plt.savefig('CTv2_dtwa/plots/min_variance_SN_vs_N_power_law_all_ranges.png'.format(interaction_range))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=None, top=None)
    plt.tight_layout()
    plt.savefig('CTv2_dtwa/plots/log_min_variance_SN_vs_log_N_power_law_all_ranges.png'.format(interaction_range))

    fig = plt.figure(figsize=(8,6))
    plt.title('power law')
    plt.xlabel('exp')
    plt.ylabel('N * <S_a^2> / <S_x>^2')
    color_idx = np.linspace(1. / len(range_list), 1., len(range_list))
    for method in ['CT', 'CTv2']:
        min_variance_SN_list = []
        for i, interaction_range in zip(color_idx, range_list):
        
            N = 200
            
            J_list = [1.]
            for J in J_list: 
                    
                dirname = method + '_dtwa'
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                if method == 'CT':
                    color = plt.cm.get_cmap('Reds')(i)
                    linestyle = 'solid'
                else:
                    color = plt.cm.get_cmap('Blues')(i)
                    linestyle = 'dashed'

                observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                print('N = {}, exp = {}, t_opt = {}'.format(N, interaction_range, t[np.argmin(variance_SN_t)]))
                min_variance_SN_list.append(min(variance_SN_t))
            
           
        plt.plot(range_list, min_variance_SN_list, 'o', label=method)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.ylim(bottom=0., top=1.)
    plt.savefig('CTv2_dtwa/plots/min_variance_SN_vs_range_power_law_N_200.png'.format(interaction_range))
    plt.ylim(bottom=1./N, top=1.)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CTv2_dtwa/plots/log_min_variance_SN_vs_range_power_law_N_200.png'.format(interaction_range))

    fig = plt.figure(figsize=(8,6))
    plt.title('power law')
    plt.xlabel('exp')
    plt.ylabel('<S_a^2> (normalized)')
    color_idx = np.linspace(1. / len(range_list), 1., len(range_list))
    for method in ['CT', 'CTv2']:
        min_variance_norm_list = []
        for i, interaction_range in zip(color_idx, range_list):
        
            N = 200
            
            J_list = [1.]
            for J in J_list: 
                    
                dirname = method + '_dtwa'
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                if method == 'CT':
                    color = plt.cm.get_cmap('Reds')(i)
                    linestyle = 'solid'
                else:
                    color = plt.cm.get_cmap('Blues')(i)
                    linestyle = 'dashed'

                observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                variance_SN_t, variance_norm_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['t']
                min_variance_norm_list.append(variance_norm_t[np.argmin(variance_SN_t)])
            
           
        plt.plot(range_list, min_variance_norm_list, 'o', label=method)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.ylim(bottom=0., top=1.)
    plt.savefig('CTv2_dtwa/plots/min_variance_norm_vs_range_power_law_N_200.png'.format(interaction_range))
    plt.ylim(bottom=1./N, top=1.)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CTv2_dtwa/plots/log_min_variance_norm_vs_range_power_law_N_200.png'.format(interaction_range))

    fig = plt.figure(figsize=(8,6))
    plt.title('power law')
    plt.xlabel('exp')
    plt.ylabel('<S_x>^2 (normalized)')
    color_idx = np.linspace(1. / len(range_list), 1., len(range_list))
    for method in ['CT', 'CTv2']:
        noise_list = []
        for i, interaction_range in zip(color_idx, range_list):
        
            N = 200
            
            J_list = [1.]
            for J in J_list: 
                    
                dirname = method + '_dtwa'
                filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                if method == 'CT':
                    color = plt.cm.get_cmap('Reds')(i)
                    linestyle = 'solid'
                else:
                    color = plt.cm.get_cmap('Blues')(i)
                    linestyle = 'dashed'

                observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                variance_SN_t, Sx, t = observed_t['min_variance_SN'], observed_t['S_x'], observed_t['t']
                noise_list.append(np.power(Sx[np.argmin(variance_SN_t)] / Sx[0], 2))

        plt.plot(range_list, noise_list, 'o', label=method)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    plt.tight_layout()
    # plt.ylim(bottom=0., top=1.)
    plt.savefig('CTv2_dtwa/plots/signal_vs_range_power_law_N_200.png'.format(interaction_range))
    # plt.ylim(bottom=1./N, top=1.)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('CTv2_dtwa/plots/log_signal_vs_log_range_power_law_N_200.png'.format(interaction_range))

    
    for interaction_range in [0,0.5,1,1.5,2,2.5,3]:
        t_opts = t_opts_vs_range[interaction_range]
        fig = plt.figure(figsize=(8,6))
        plt.title('power law, exp = {}'.format(interaction_range))
        plt.xlabel('N')
        plt.ylabel('t_opt')
        N_list = [10,20,50,100,200]
        for method in ['CT', 'CTv2']:
            t_opt_list = t_opts[method]
            prev = plt.plot(N_list, t_opt_list, 'o', label=method)
            prev_color = prev[-1].get_color()
            popt, pcov = curve_fit(fn, np.array(N_list)[2:], t_opt_list[2:])
            plt.plot(N_list, fn(N_list, *popt), label='{} / (N ^ {})'.format(round(popt[0],2),round(popt[1],2)), color=prev_color, linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
        plt.tight_layout()
        plt.savefig('CTv2_dtwa/plots/t_opt_vs_N_power_law_exp_{}.png'.format(interaction_range))
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('CTv2_dtwa/plots/log_t_opt_vs_log_N_power_law_exp_{}.png'.format(interaction_range))

    
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
                    dirname = method + '_dtwa'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                    plt.plot(t, variance_SN_t, label='{}, exp = {}'.format(method, interaction_range))
        plt.ylim(bottom=0., top=1.)
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('CTv2_dtwa/plots/variance_SN_vs_t_N_{}_power_law_all_ranges.png'.format(N))


    N_list = [200]
    for i, N in zip(color_idx, N_list):
        fig = plt.figure()
        plt.title('N = {}, power law'.format(N))
        plt.xlabel('t')
        plt.ylabel('N * <S_a^2> / <S_x>^2')
        for interaction_range in [0, 1, 3]:
            J_list = [1.]
            for J in J_list: 
                for method in ['CT', 'CTv2']:
                    dirname = method + '_dtwa'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_SN_t, Sx, t = observed_t['min_variance_SN'], observed_t['S_x'], observed_t['t']
                    linestyle = 'solid' if method == 'CT' else 'dashed'
                    plt.plot(t, variance_SN_t, label='{}, exp = {}'.format(method, interaction_range), linestyle=linestyle)
        plt.yscale('log')
        plt.ylim(bottom=1./N, top=1.)
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('CTv2_dtwa/plots/variance_SN_vs_t_N_{}_power_law_all_ranges.png'.format(N))

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
                    dirname = method + '_dtwa'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_norm_t, Sx, t = observed_t['min_variance_norm'], observed_t['S_x'], observed_t['t']
                    linestyle = 'solid' if method == 'CT' else 'dashed'
                    plt.plot(t, variance_norm_t, label='{}, exp = {}'.format(method, interaction_range), linestyle=linestyle)
        plt.ylim(bottom=1./N, top=1.)
        plt.yscale('log')
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('CTv2_dtwa/plots/variance_norm_vs_t_N_{}_power_law_all_ranges.png'.format(N))

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
                    dirname = method + '_dtwa'
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
                    if method == 'CT':
                        color = plt.cm.get_cmap('Reds')(i)
                        linestyle = 'solid'
                    else:
                        color = plt.cm.get_cmap('Blues')(i)
                        linestyle = 'dashed'
                    observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                    variance_norm_t, Sx, t = observed_t['min_variance_norm'], observed_t['S_x'], observed_t['t']
                    linestyle = 'solid' if method == 'CT' else 'dashed'
                    plt.plot(t, np.power(Sx / Sx[0], 2), label='{}, exp = {}'.format(method, interaction_range), linestyle=linestyle)
        plt.ylim(bottom=1./N, top=1.)
        plt.yscale('log')
        plt.xlim(left=0., right=1.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('CTv2_dtwa/plots/signal_vs_t_N_{}_power_law_all_ranges.png'.format(N))



