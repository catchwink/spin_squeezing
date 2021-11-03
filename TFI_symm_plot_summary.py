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
    
    dirname = 'TFI_symm'

    variance_SN_vs_h_vs_range_vs_N = defaultdict(dict)
    t_vs_h_vs_range_vs_N = defaultdict(dict)
    method = 'TFI'

    variance_SN_vs_range_vs_N_h_1 = defaultdict(dict)
    t_vs_range_vs_N_h_1 = defaultdict(dict)
    variance_SN_vs_N_vs_range_h_1 = defaultdict(dict)
    t_vs_N_vs_range_h_1 = defaultdict(dict)

    variance_SN_vs_range_vs_N_h_m1 = defaultdict(dict)
    t_vs_range_vs_N_h_m1 = defaultdict(dict)
    variance_SN_vs_N_vs_range_h_m1 = defaultdict(dict)
    t_vs_N_vs_range_h_m1 = defaultdict(dict)

    for Jz in [-1]:
        for N in [10, 20, 50, 100, 200, 500, 1000]:
            spin_system = sd.SpinOperators_Symmetry((N,1))
            variance_SN_vs_h_vs_range = defaultdict(dict)
            t_vs_h_vs_range = defaultdict(dict)
            # for interaction_range in [0,0.5,1,1.5,2,2.5,3]:
            for interaction_range in [0]:
                variance_SN_vs_h = {}
                t_vs_h = {}
                # h_list = N * Jz * np.array([-2., -1., -0.75, -0.5, -0.25, -0.125, -0.1, -0.05, -0.04, -0.03, -0.025, -0.02, -0.0125, -0.01, 0, 0.01, 0.0125, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.125, 0.25, 0.5, 0.75, 1., 2.])
                # for h in h_list: 
                h_list1 = spin_system.N * Jz * np.array([-2., -1., -0.75, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 0.75, 1., 2.])
                h_list2 = spin_system.N * Jz * np.array([-0.1, -0.05, -0.025, -0.0125, 0.0125, 0.025, 0.05, 0.1])
                h_list3 = spin_system.N * Jz * np.array([-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04])
                h_list4 = np.arange(-10,10.5,0.5)
                # h_list4 = np.array([-1., 1., -2.5, 2.5])
                # h_list5 = np.array([-3., 3., -3.5, 3.5, -4., 4., -4.5, 4.5])
                for h in sorted(np.concatenate([h_list1, h_list2, h_list3, h_list4])):
                    filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_Jz_{}_h_{}'.format(method, N, 'power_law', 'exp', interaction_range, Jz, h)
                    if filename in os.listdir(dirname): 
                        observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                        variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)
                        t = observed_t['t']

                        variance_SN_vs_h[h] = variance_SN_t
                        t_vs_h[h] = t

                        if h == 1.0:
                            variance_SN_vs_range_vs_N_h_1[N][interaction_range] = variance_SN_t
                            t_vs_range_vs_N_h_1[N][interaction_range] = t
                            variance_SN_vs_N_vs_range_h_1[interaction_range][N] = variance_SN_t
                            t_vs_N_vs_range_h_1[interaction_range][N] = t
                        if h == -1.0:
                            variance_SN_vs_range_vs_N_h_m1[N][interaction_range] = variance_SN_t
                            t_vs_range_vs_N_h_m1[N][interaction_range] = t
                            variance_SN_vs_N_vs_range_h_m1[interaction_range][N] = variance_SN_t
                            t_vs_N_vs_range_h_m1[interaction_range][N] = t

                variance_SN_vs_h_vs_range[interaction_range] = variance_SN_vs_h
                t_vs_h_vs_range[interaction_range] = t_vs_h
            variance_SN_vs_h_vs_range_vs_N[N] = variance_SN_vs_h_vs_range
            t_vs_h_vs_range_vs_N[N] = t_vs_h_vs_range

        N_list = []
        optimal_h_scaled = []
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('h / NJ')
            plt.ylabel('N * <S_a^2> / <S_x>^2')
            plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                min_variance_SN_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    h_list.append(h)
                    min_variance_SN_list.append(min(variance_SN_t))
                plt.plot(np.array(h_list) / (N * Jz), min_variance_SN_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                if interaction_range == 0:
                    N_list.append(N)
                    optimal_h_scaled.append((np.array(h_list) / (N * Jz))[np.argmin(min_variance_SN_list)])
            plt.legend()
            plt.savefig('{}/plots/min_variance_SN_vs_h_scaled_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))
            plt.ylim(bottom=0.9 * min(min_variance_SN_list), top=1.1 * max(min_variance_SN_list))
            plt.xlim(left=-5/(N * np.abs(Jz)), right=5/(N * np.abs(Jz)))
            plt.savefig('{}/plots/min_variance_SN_vs_h_scaled_N_{}_Jz_{}_all_ranges_zoom.png'.format(dirname, N, Jz))

        fig = plt.figure(figsize=(7.2,4.8))
        title = 'power law, exp = 0, Jz = {}'.format(Jz)
        plt.title(title)
        plt.xlabel('N')
        plt.ylabel('h / NJ')
        plt.plot(N_list, optimal_h_scaled, 'o')
        plt.savefig('{}/plots/opt_h_scaled_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        N_list = []
        optimal_h_scaled_for_rate = []
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('h / NJ')
            plt.ylabel('- Δ(N * <S_a^2> / <S_x>^2) / Δt')
            # plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                variance_SN_rate_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    h_list.append(h)
                    t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                    init_rate = (variance_SN_t[0] - variance_SN_t[1]) / (t[1] - t[0])
                    variance_SN_rate_list.append(init_rate)
                plt.plot(np.array(h_list) / (N * Jz), variance_SN_rate_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                if interaction_range == 0:
                    N_list.append(N)
                    optimal_h_scaled_for_rate.append((np.array(h_list) / (N * Jz))[np.argmax(variance_SN_rate_list)])
            plt.legend()
            # plt.ylim(bottom=1.5 * min(min_variance_SN_rate_list), top=0.5 * max(min_variance_SN_rate_list))
            plt.savefig('{}/plots/init_rate_v0_variance_SN_vs_h_scaled_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))


        fig = plt.figure(figsize=(7.2,4.8))
        title = 'power law, exp = 0, Jz = {}'.format(Jz)
        plt.title(title)
        plt.xlabel('N')
        plt.ylabel('h / NJ')
        plt.plot(N_list, optimal_h_scaled_for_rate, 'o')
        plt.savefig('{}/plots/opt_h_scaled_for_init_rate_v0_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        N_list = []
        optimal_h_scaled_for_rate = []
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('h / NJ')
            plt.ylabel('- Δ(N * <S_a^2> / <S_x>^2) / Δt')
            # plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                variance_SN_rate_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    h_list.append(h)
                    t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                    init_rate = (1 - np.min(variance_SN_t)) / t[np.argmin(variance_SN_t)]
                    variance_SN_rate_list.append(init_rate)
                plt.plot(np.array(h_list) / (N * Jz), variance_SN_rate_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                if interaction_range == 0:
                    N_list.append(N)
                    optimal_h_scaled_for_rate.append((np.array(h_list) / (N * Jz))[np.argmax(variance_SN_rate_list)])
            plt.legend()
            # plt.ylim(bottom=1.5 * min(min_variance_SN_rate_list), top=0.5 * max(min_variance_SN_rate_list))
            plt.savefig('{}/plots/init_rate_v1_variance_SN_vs_h_scaled_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))


        fig = plt.figure(figsize=(7.2,4.8))
        title = 'power law, exp = 0, Jz = {}'.format(Jz)
        plt.title(title)
        plt.xlabel('N')
        plt.ylabel('h / NJ')
        plt.plot(N_list, optimal_h_scaled_for_rate, 'o')
        plt.savefig('{}/plots/opt_h_scaled_for_init_rate_v1_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        N_list = []
        optimal_h_scaled_for_rate = []
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('h / NJ')
            plt.ylabel('- Δ(N * <S_a^2> / <S_x>^2) / Δt')
            # plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                variance_SN_rate_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    h_list.append(h)
                    t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                    stop_idx = np.argmin(variance_SN_t[t<=0.5]) + 1
                    coeffs = np.polyfit(t[:stop_idx], variance_SN_t[:stop_idx], 2) # quadratic, linear, intercept
                    variance_SN_rate_list.append(- list(coeffs)[1])
                plt.plot(np.array(h_list) / (N * Jz), variance_SN_rate_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                if interaction_range == 0:
                    N_list.append(N)
                    optimal_h_scaled_for_rate.append((np.array(h_list) / (N * Jz))[np.argmax(variance_SN_rate_list)])
            plt.legend()
            # plt.ylim(bottom=1.5 * min(min_variance_SN_rate_list), top=0.5 * max(min_variance_SN_rate_list))
            plt.savefig('{}/plots/init_rate_v2_variance_SN_vs_h_scaled_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))


        fig = plt.figure(figsize=(7.2,4.8))
        title = 'power law, exp = 0, Jz = {}'.format(Jz)
        plt.title(title)
        plt.xlabel('N')
        plt.ylabel('h / NJ')
        plt.plot(N_list, optimal_h_scaled_for_rate, 'o')
        plt.savefig('{}/plots/opt_h_scaled_for_init_rate_v2_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        N_list = []
        optimal_h = []
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('h')
            plt.ylabel('N * <S_a^2> / <S_x>^2')
            plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                min_variance_SN_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    h_list.append(h)
                    min_variance_SN_list.append(min(variance_SN_t))
                plt.plot(h_list, min_variance_SN_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                if interaction_range == 0:
                    N_list.append(N)
                    optimal_h.append(np.array(h_list)[np.argmin(min_variance_SN_list)])
            plt.legend()
            plt.savefig('{}/plots/min_variance_SN_vs_h_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))
            plt.ylim(bottom=0.9 * min(min_variance_SN_list), top=1.1 * max(min_variance_SN_list))
            plt.xlim(left=-5, right=5)
            plt.yscale('log')
            plt.savefig('{}/plots/min_variance_SN_vs_h_N_{}_Jz_{}_all_ranges_zoom.png'.format(dirname, N, Jz))

        fig = plt.figure(figsize=(7.2,4.8))
        title = 'power law, exp = 0, Jz = {}'.format(Jz)
        plt.title(title)
        plt.xlabel('N')
        plt.ylabel('h')
        plt.plot(N_list, optimal_h, 'o')
        plt.savefig('{}/plots/opt_h_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        N_list = []
        optimal_h_for_rate = []
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('h')
            plt.ylabel('- Δ(N * <S_a^2> / <S_x>^2) / Δt')
            # plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                variance_SN_rate_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    h_list.append(h)
                    t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                    init_rate = (variance_SN_t[0] - variance_SN_t[1]) / (t[1] - t[0])
                    variance_SN_rate_list.append(init_rate)
                plt.plot(h_list, variance_SN_rate_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                if interaction_range == 0:
                    N_list.append(N)
                    optimal_h_for_rate.append(np.array(h_list)[np.argmax(variance_SN_rate_list)])
                plt.legend()
            plt.savefig('{}/plots/init_rate_v0_variance_SN_vs_h_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))

        fig = plt.figure(figsize=(7.2,4.8))
        title = 'power law, exp = 0, Jz = {}'.format(Jz)
        plt.title(title)
        plt.xlabel('N')
        plt.ylabel('h')
        plt.plot(N_list, optimal_h_for_rate, 'o')
        plt.savefig('{}/plots/opt_h_for_init_rate_v0_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        N_list = []
        optimal_h_for_rate = []
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('h')
            plt.ylabel('- Δ(N * <S_a^2> / <S_x>^2) / Δt')
            # plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                variance_SN_rate_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    h_list.append(h)
                    t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                    init_rate = (1 - np.min(variance_SN_t)) / t[np.argmin(variance_SN_t)]
                    variance_SN_rate_list.append(init_rate)
                plt.plot(h_list, variance_SN_rate_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                if interaction_range == 0:
                    N_list.append(N)
                    optimal_h_for_rate.append(np.array(h_list)[np.argmax(variance_SN_rate_list)])
                plt.legend()
            plt.savefig('{}/plots/init_rate_v1_variance_SN_vs_h_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))

        fig = plt.figure(figsize=(7.2,4.8))
        title = 'power law, exp = 0, Jz = {}'.format(Jz)
        plt.title(title)
        plt.xlabel('N')
        plt.ylabel('h')
        plt.plot(N_list, optimal_h_for_rate, 'o')
        plt.savefig('{}/plots/opt_h_for_init_rate_v1_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        N_list = []
        optimal_h_for_rate = []
        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('h')
            plt.ylabel('- Δ(N * <S_a^2> / <S_x>^2) / Δt')
            # plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                h_list = []
                variance_SN_rate_list = []
                for h, variance_SN_t in variance_SN_vs_h.items():
                    h_list.append(h)
                    t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                    stop_idx = np.argmin(variance_SN_t[t<=0.5]) + 1
                    coeffs = np.polyfit(t[:stop_idx], variance_SN_t[:stop_idx], 2) # quadratic, linear, intercept
                    variance_SN_rate_list.append(- list(coeffs)[1])
                plt.plot(h_list, variance_SN_rate_list, marker='o', label='exp = {}'.format(interaction_range), color=plt.cm.get_cmap('Reds')(i))
                if interaction_range == 0:
                    N_list.append(N)
                    optimal_h_for_rate.append(np.array(h_list)[np.argmax(variance_SN_rate_list)])
                plt.legend()
            plt.savefig('{}/plots/init_rate_v2_variance_SN_vs_h_N_{}_Jz_{}_all_ranges.png'.format(dirname, N, Jz))

        fig = plt.figure(figsize=(7.2,4.8))
        title = 'power law, exp = 0, Jz = {}'.format(Jz)
        plt.title(title)
        plt.xlabel('N')
        plt.ylabel('h')
        plt.plot(N_list, optimal_h_for_rate, 'o')
        plt.savefig('{}/plots/opt_h_for_init_rate_v2_vs_N_Jz_{}_all_ranges.png'.format(dirname, Jz))
        plt.close()

        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            ax = fig.add_subplot(111, projection='3d')
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            ax.set_title(title)
            ax.set_xlabel('h / NJ')
            ax.set_ylabel('t')
            ax.set_zlabel('N * <S_a^2> / <S_x>^2')
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                color_idx = np.linspace(1. / len(variance_SN_vs_h), 1., len(variance_SN_vs_h))
                for j, (h, variance_SN_t) in zip(color_idx, variance_SN_vs_h.items()):
                    t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                    variance_SN_t_masked = np.array([v if v < 1. else np.nan for v in variance_SN_t])
                    ax.plot(np.full(t.shape, h / (N * Jz)), t, variance_SN_t_masked, label='h / NJ = {}'.format(h / (N * Jz)), color=plt.cm.get_cmap('coolwarm')(j))
            ax.set_zlim(bottom=0., top=1.)
            ax.set_ylim(bottom=0., top=0.2)
            plt.savefig('{}/plots/variance_SN_vs_t_3d_N_{}_power_law_exp_0_Jz_{}_all_h.png'.format(dirname, N, Jz))

        for N, variance_SN_vs_h_vs_range in variance_SN_vs_h_vs_range_vs_N.items():
            fig = plt.figure(figsize=(7.2,4.8))
            title = 'N = {}, power law, Jz = {}'.format(N, Jz)
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('N * <S_a^2> / <S_x>^2')
            # plt.xlim(left=0., right=0.04)
            # plt.ylim(bottom=0., top=1.)
            color_idx = np.linspace(1. / len(variance_SN_vs_h_vs_range), 1., len(variance_SN_vs_h_vs_range))
            for i, (interaction_range, variance_SN_vs_h) in zip(color_idx, variance_SN_vs_h_vs_range.items()):
                color_idx = np.linspace(1. / len(variance_SN_vs_h), 1., len(variance_SN_vs_h))
                for j, (h, variance_SN_t) in zip(color_idx, variance_SN_vs_h.items()):
                    t = t_vs_h_vs_range_vs_N[N][interaction_range][h]
                    plt.plot(t, variance_SN_t, label='h / NJ = {}'.format(h / (N * Jz)), color=plt.cm.get_cmap('coolwarm')(j))
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.), prop={'size': 6})
            plt.tight_layout()
            plt.yscale('log')
            plt.ylim(top=1., bottom=0.01)
            plt.savefig('{}/plots/variance_SN_vs_t_N_{}_power_law_exp_0_Jz_{}_all_h.png'.format(dirname, N, Jz))
