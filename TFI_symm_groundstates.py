import numpy as np
import setup
import spin_dynamics as sd
import util
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()

    # only uniform all-to-all interactions in symmetry
    if interaction_shape == 'power_law':
        interaction_range_list = [0]
    else:
        interaction_range_list = [max(interaction_range_list)]

    method = 'TFI'
    dirname = 'TFI_symm_gs'
    spin_system = sd.SpinOperators_Symmetry(system_size)
    N = spin_system.N
    observables = spin_system.get_observables()
    for interaction_range in interaction_range_list:
        for Jz in [1., -1.]:
            h_list = N * np.array([-10., -2., -1.5, -1., -0.5, -0.1, -0.01, 0., 0.01, 0.1, 0.5, 1., 1.5, 2., 10.])
            variance_SN_vs_h = []
            gsE_vs_h = []
            es1E_vs_h = []
            es2E_vs_h = []
            for h in h_list:
                ham_terms = ['Sz_sq', 'S_x']
                strengths = [Jz, h]
                H = spin_system.get_Hamiltonian(ham_terms, strengths)
                eigvals, eigvecs = eigsh(H, k=3, which='SA')
                gsE, gs = eigvals[0], eigvecs[:,0]
                es1E, es1 = eigvals[1], eigvecs[:,1]
                es2E, es2 = eigvals[2], eigvecs[:,2]
                observed_t = {}
                sd.store_observables(observables, observed_t, [gs], t_it=[0])

                variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)

                variance_SN_vs_h.append(variance_SN_t[0])
                gsE_vs_h.append(gsE)
                es1E_vs_h.append(es1E)
                es2E_vs_h.append(es2E)

            fig = plt.figure()
            plt.title('TFI, N = {}, power_law, exp=0, J_z = {}'.format(N, Jz))
            plt.ylabel('N * <S_a^2> / <S_x>^2')
            plt.xlabel('h')
            plt.plot(h_list, variance_SN_vs_h)
            plt.ylim(bottom=0., top=1.)
            plt.savefig('{}/variance_SN_vs_h_N_{}_{}_{}_{}_Jz_{}.pn  g'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range, Jz))

            fig = plt.figure()
            plt.title('TFI, N = {}, power_law, exp=0, J_z = {}'.format(N, Jz))
            plt.ylabel('ΔE')
            plt.xlabel('h')
            ground_state_energy = np.array(gsE_vs_h)
            excited_state_energy = []
            for i in range(len(es1E_vs_h)):
                if round(gsE_vs_h[i], 8) == round(es1E_vs_h[i], 8):
                    excited_state_energy.append(es2E_vs_h[i])
                else:
                    excited_state_energy.append(es1E_vs_h[i])
            print(np.array(es1E_vs_h) - np.array(gsE_vs_h))
            plt.plot(h_list, np.array(excited_state_energy) - np.array(ground_state_energy), label='ΔE', marker='o')
            plt.plot(h_list, np.array(es1E_vs_h) - np.array(gsE_vs_h), label='E_1 - E_0', marker='o', linestyle='dotted')
            plt.plot(h_list, np.array(es2E_vs_h) - np.array(es1E_vs_h), label='E_2 - E_1', marker='o', linestyle='dotted')

            plt.vlines([-N, N], 1e-8, 2*N, label='±N', linestyle='dashed')
            plt.legend()
            plt.ylim(bottom=1e-8, top=2*N)
            plt.tight_layout()
            plt.savefig('{}/energy_gap_vs_h_N_{}_{}_{}_{}_Jz_{}.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range, Jz))
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig('{}/log_energy_gap_vs_h_N_{}_{}_{}_{}_Jz_{}.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range, Jz))
            
            plt.xlim(-1.5 * N, 1.5 * N)
            plt.tight_layout()
            plt.savefig('{}/energy_gap_vs_h_zoom_N_{}_{}_{}_{}_Jz_{}.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range, Jz))
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig('{}/log_energy_gap_vs_h_zoom_N_{}_{}_{}_{}_Jz_{}.png'.format(dirname, N, interaction_shape, interaction_param_name, interaction_range, Jz))


            print(h_list[np.argmin(variance_SN_vs_h)])