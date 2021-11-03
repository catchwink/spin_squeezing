# import numpy as np
# import setup
# import spin_dynamics as sd
# import util

# if __name__ == '__main__':
#     np.set_printoptions(threshold=np.inf)

#     system_size = (4,1)
#     spin_system = sd.SpinOperators_Symmetry(system_size)

#     ham_terms = ['Sz_sq', 'S_x']
#     strengths = [1., 0.]
#     H = spin_system.get_Hamiltonian(ham_terms, strengths)
#     print('TFI, h=0: ', H)

#     ham_terms = ['Sz_sq']
#     strengths = [1.]
#     H = spin_system.get_Hamiltonian(ham_terms, strengths)
#     print('ZZ: ', H)

#     ham_terms = ['Sx_sq', 'Sy_sq']
#     strengths = [1., 1.]
#     H = spin_system.get_Hamiltonian(ham_terms, strengths)
#     print('XY: ', H)

#     J_eff = -1
#     Jz = (J_eff + 1.) / 2
#     Jperp = 2 * (1 - Jz)
#     ham_terms = ['Sz_sq', 'Sx_sq', 'Sy_sq']
#     strengths = [Jz, Jperp/2., Jperp/2.]
#     H = spin_system.get_Hamiltonian(ham_terms, strengths)
#     print('XXZ, J_eff=-1: ', H)

#     J_eff = 1
#     Jz = (J_eff + 1.) / 2
#     Jperp = 2 * (1 - Jz)
#     ham_terms = ['Sz_sq', 'Sx_sq', 'Sy_sq']
#     strengths = [Jz, Jperp/2., Jperp/2.]
#     H = spin_system.get_Hamiltonian(ham_terms, strengths)
#     print('XXZ, J_eff=1: ', H)

import experiment_realistic
import numpy as np
import setup
import spin_dynamics as sd
import util
from matplotlib import pyplot as plt



for N in [5,10,20,50,100]:
    
    # min_variance_SN_list = []
    for _ in range(10):
        fig = plt.figure()
        plt.xlabel('t')
        plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
        plt.title('N = {}, Gaussian distribution, Rydberg dressing'.format(N))
        min_overall = 1.
        experiment = experiment_realistic.SqueezingCalculations(interaction_type='RD', N_atoms=N, detuning=16e6, rabi_freq=1.1e6, C_6=100e9)

        spin_system = sd.SpinOperators_DTWA('square', (N,1), 1., coord=experiment.points)

        psi_0 = spin_system.get_init_state('x')
        J = 1.
        
        
        H = spin_system.get_CT_Hamiltonian(J, [], Jij=-experiment.interactionMatrix)
        
        spin_evolution = sd.SpinEvolution(H, psi_0)
        t_max = 1.
        t = np.linspace(t_max/100, t_max, 100)
        tdist, t = spin_evolution.evolve([1.], t, store_states=True)
        meanConfig_evol = np.mean(tdist,axis=1)
                
        min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)

        min_overall = min(min_overall, min(min_variance_SN_t))

        plt.plot(t, min_variance_SN_t, label='CT')

        H = spin_system.get_Ising_Hamiltonian(J, [], Jij=-experiment.interactionMatrix)
        
        spin_evolution = sd.SpinEvolution(H, psi_0)
        t_max = 1.
        t = np.linspace(t_max/100, t_max, 100)
        tdist, t = spin_evolution.evolve([1.], t, store_states=True)
        meanConfig_evol = np.mean(tdist,axis=1)
                
        min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)

        min_overall = min(min_overall, min(min_variance_SN_t))

        plt.plot(t, min_variance_SN_t, label='ZZ')
        plt.legend()
        # min_variance_SN_list.append(min_variance_SN_t)
    # plt.errorbar(t, np.mean(min_variance_SN_list, axis=0), yerr=np.std(min_variance_SN_list, axis=0), label='N = {}, ZZ'.format(N))
        plt.ylim(bottom=0.8 * min_overall, top=1.2)
        # plt.legend()
        plt.savefig('test_inhomogeneous_N_{}_t_{}.png'.format(N, _))
        print('t = {}, coords = '.format(_))
        print(experiment.points)
        plt.close()
    # results_t = spin_system.get_observed(tdist, meanConfig_evol)
    # results_t['min_variance_SN'] = min_variance_SN_t
    # results_t['min_variance_norm'] = min_variance_norm_t
    # results_t['opt_angle'] = opt_angle_t
    # results_t['t'] = t

    # util.store_observed_t(results_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J))