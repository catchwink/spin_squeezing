import numpy as np
import setup
import spin_dynamics as sd
import util
from sympy.physics.quantum.cg import CG
from scipy.linalg import expm
from matplotlib import pyplot as plt
import matplotlib
import csv
import imageio
import os

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range_list, instance = setup.configure()

    # only uniform all-to-all interactions in symmetry
    if interaction_shape == 'power_law':
        interaction_range_list = [0]
    else:
        interaction_range_list = [max(interaction_range_list)]

    method = 'TFI'
    spin_system = sd.SpinOperators_Symmetry(system_size)
    observables = spin_system.get_observables()
    psi_0 = spin_system.get_init_state('x')

    c_Sm = {}
    S = spin_system.N // 2
    for m in range(-S, S+1):
        c_Sm[m] = np.sum([np.sqrt((2 * k + 1) / (4 * np.pi)) * float(CG(S,m,S,-m,k,0).doit()) for k in range(2 * S + 1)])

    Ryz = expm((- 1j * np.pi / 2) * observables['S_x'].tocsc())
    Rzy = expm((1j * np.pi / 2) * observables['S_x'].tocsc())
    Rxz = expm((1j * np.pi / 2) * observables['S_y'].tocsc())
    Rzx = expm((- 1j * np.pi / 2) * observables['S_y'].tocsc())
    def Rz(phi):
        phi = float(phi)
        return expm((1j * phi) * observables['S_z'].tocsc())
    def rotated(v, theta, phi):
        return Rzy @ Rz(theta) @ Ryz @ Rz(phi) @ v

    for interaction_range in interaction_range_list:
        Jz = -1
        # h_list1 = spin_system.N * Jz * np.array([-2., -1., -0.75, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 0.75, 1., 2.])
        # h_list2 = spin_system.N * Jz * np.array([-0.1, -0.05, -0.025, -0.0125, 0.0125, 0.025, 0.05, 0.1])
        # h_list3 = spin_system.N * Jz * np.array([-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04])
        # for h in np.concatenate([h_list1, h_list2, h_list3]):
        for h in spin_system.N * Jz * np.array([-2., 2., 0.5]):
            ham_terms = ['Sz_sq', 'S_x']
            strengths = [Jz, h]
            H = spin_system.get_Hamiltonian(ham_terms, strengths)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            t_max = 0.2
            t = np.linspace(t_max/40., t_max, 40)
            psi_t, observed_t, t = spin_evolution.evolve([1.], t, observables=observables, store_states=True)
            variance_SN_t, variance_norm_t, angle_t = spin_system.get_squeezing(observed_t)

            opt_t = t[np.argmin(variance_SN_t)]
            # t_select = [t[np.argmin(variance_SN_t) // 10 * i] for i in range(10)] + [opt_t]
            # psi_select = [psi_t[np.argmin(variance_SN_t) // 10 * i] for i in range(10)] + [psi_t[np.argmin(variance_SN_t)]]
            t_select = t[:np.argmin(variance_SN_t)]
            psi_select = psi_t[:np.argmin(variance_SN_t)]
            
            fig = plt.figure()
            plt.plot(t, variance_SN_t)
            plt.savefig('TFI_symm_wigner_2/variance_SN_vs_t_N_{}_{}_{}_{}_Jz_{}_h_{}.png'.format(spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h))
            plt.close()
            theta = np.linspace(0, np.pi, num=(5*S+1))  
            phi = np.linspace(0, 2 * np.pi, num=(10*S+1))
            theta, phi = np.meshgrid(theta, phi)
            x, y, z = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

            def wigner_dist(v, t, p):
                ro_diag = np.square(np.abs(rotated(v, t, p)))
                m_range = list(reversed(range(-S, S+1)))
                return np.sum([c_Sm[m_range[i]] * ((-1)**(S-m_range[i])) * ro_diag[i] for i in range(len(m_range))])

            wigner_vs_t = []
            ro_vs_t = []
            filenames = []
            for t_cur, psi in zip(t_select, psi_select):
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.set_title('N = {}, {}, {} = {}, Jz = {}, h = {}'.format(spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(20, 10)
                wigner_rows = []
                for th, ph in zip(theta, phi):
                    wigner_cols = []
                    for t, p in zip(th, ph):
                        wigner_cols.append(wigner_dist(psi, t, p))
                    wigner_rows.append(wigner_cols)
                wigner = np.array(wigner_rows)
                wigner = wigner * spin_system.N * np.sqrt(2 * S / np.pi)
                norm = matplotlib.colors.Normalize(0, spin_system.N * np.sqrt(2 * S / np.pi))
                scalarmappable = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
                scalarmappable.set_array([])
                fcolors = scalarmappable.to_rgba(wigner)
                surf = ax.plot_surface(x, y, z, facecolors=fcolors)
                fig.colorbar(scalarmappable)
                filename = 'TFI_symm_wigner_2/wigner_N_{}_{}_{}_{}_Jz_{}_h_{}_t_{}.png'.format(spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h, t_cur)
                plt.savefig(filename)
                plt.close()
                wigner_vs_t.append(wigner)
                ro_vs_t.append(np.square(np.abs(psi)))
                filenames.append(filename)

            with imageio.get_writer('TFI_symm_wigner_2/wigner_N_{}_{}_{}_{}_Jz_{}_h_{}.gif'.format(spin_system.N, interaction_shape, interaction_param_name, interaction_range, Jz, h), mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    os.remove(filename)

                    

