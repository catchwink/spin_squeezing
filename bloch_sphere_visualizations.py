import numpy as np
import setup
import spin_dynamics as sd
import util
from DTWA.DTWA_Lib import visualizeBloch, animateBloch

if __name__ == '__main__':
    elev,azim = (20,-40)
    np.set_printoptions(threshold=np.inf)
    interaction_shape, interaction_param_name = 'power_law', 'exp'
    system_size = (4, 1)
    structure, fill = 'square', 1.0

    method = 'ZZ'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    psi_0 = spin_system.get_init_state('x')
    for interaction_range in [0]:
        J = 1.
        H = spin_system.get_Ising_Hamiltonian(J, interaction_range)
        spin_evolution = sd.SpinEvolution(H, psi_0)
        t_max = 4.
        # t = np.linspace(t_max/500, t_max, 500)
        t = np.linspace(t_max/10, t_max, 10 )
        tdist, t = spin_evolution.evolve([1.], t, store_states=True)
        meanConfig_evol = np.mean(tdist,axis=1)
            
        min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
        results_t = spin_system.get_observed(tdist, meanConfig_evol)
        results_t['min_variance_SN'] = min_variance_SN_t
        results_t['min_variance_norm'] = min_variance_norm_t
        results_t['opt_angle'] = opt_angle_t
        results_t['t'] = t

        util.store_observed_t(results_t, 'bloch_vis/observables_vs_t_{}_N_{}_{}_{}_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range))

        _, ax = visualizeBloch(tdist[0],t[0],viewAngle=(0,0),axesLabels=True,showProjection=False);
        ax.figure.savefig('bloch_vis/bloch_t_0_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

        _, ax = visualizeBloch(tdist[np.argmin(min_variance_SN_t)],t[np.argmin(min_variance_SN_t)],viewAngle=(0,0),axesLabels=True,showProjection=False);
        ax.figure.savefig('bloch_vis/bloch_t_opt_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

        _, ax = visualizeBloch(tdist[0],t[0],viewAngle=(elev,azim),axesLabels=True,showProjection=True);
        ax.figure.savefig('bloch_vis/bloch2_t_0_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

        _, ax = visualizeBloch(tdist[np.argmin(min_variance_SN_t)],t[np.argmin(min_variance_SN_t)],viewAngle=(elev,azim),axesLabels=True,showProjection=True);
        ax.figure.savefig('bloch_vis/bloch2_t_opt_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

        filename = 'bloch_vis/bloch_animate_{}_N_{}_{}_{}_{}.gif'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range)
        animateBloch(t,tdist,sphere=[],viewAngle=(30,-45),showProjection=True,axesLabels=True,showAxes=True,saveBool=True,filename=filename,fps=4)

    method = 'CT'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    psi_0 = spin_system.get_init_state('x')
    for interaction_range in [0]:
        for J in [1.]:
            H = spin_system.get_CT_Hamiltonian(J, interaction_range)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            t_max = 4.
            t = np.linspace(t_max/500, t_max, 500)
            tdist, t = spin_evolution.evolve([1.], t, store_states=True)
            meanConfig_evol = np.mean(tdist,axis=1)
            
            min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
            results_t = spin_system.get_observed(tdist, meanConfig_evol)
            results_t['min_variance_SN'] = min_variance_SN_t
            results_t['min_variance_norm'] = min_variance_norm_t
            results_t['opt_angle'] = opt_angle_t
            results_t['t'] = t

            util.store_observed_t(results_t, 'bloch_vis/observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J))

            _, ax = visualizeBloch(tdist[0],t[0],viewAngle=(0,0),axesLabels=True,showProjection=False);
            ax.figure.savefig('bloch_vis/bloch_t_0_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')
            
            _, ax = visualizeBloch(tdist[np.argmin(min_variance_SN_t)],t[np.argmin(min_variance_SN_t)],viewAngle=(0,0),axesLabels=True,showProjection=False);
            ax.figure.savefig('bloch_vis/bloch_t_opt_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

            _, ax = visualizeBloch(tdist[0],t[0],viewAngle=(elev,azim),axesLabels=True,showProjection=True);
            ax.figure.savefig('bloch_vis/bloch2_t_0_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

            _, ax = visualizeBloch(tdist[np.argmin(min_variance_SN_t)],t[np.argmin(min_variance_SN_t)],viewAngle=(elev,azim),axesLabels=True,showProjection=True);
            ax.figure.savefig('bloch_vis/bloch2_t_opt_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

            filename = 'bloch_vis/bloch_animate_{}_N_{}_{}_{}_{}.gif'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range)
            animateBloch(t,tdist,sphere=[],viewAngle=(30,-45),showProjection=True,axesLabels=True,showAxes=True,saveBool=True,filename=filename,fps=4)


    method = 'TFI'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    psi_0 = spin_system.get_init_state('x')
    for interaction_range in [0]:
        Jz = 1
        h_list = [-1.]
        for h in h_list:
            H = spin_system.get_TFI_Hamiltonian(Jz, h, interaction_range)
            spin_evolution = sd.SpinEvolution(H, psi_0)
            t_max = 4.
            t = np.linspace(t_max/500, t_max, 500)
            tdist, t = spin_evolution.evolve([1.], t, store_states=True)
            meanConfig_evol = np.mean(tdist,axis=1)
            
            min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
            results_t = spin_system.get_observed(tdist, meanConfig_evol)
            results_t['min_variance_SN'] = min_variance_SN_t
            results_t['min_variance_norm'] = min_variance_norm_t
            results_t['opt_angle'] = opt_angle_t
            results_t['t'] = t

            util.store_observed_t(results_t, 'bloch_vis/observables_vs_t_{}_N_{}_{}_{}_{}_h_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, h))

            _, ax = visualizeBloch(tdist[0],t[0],viewAngle=(0,0),axesLabels=True,showProjection=False);
            ax.figure.savefig('bloch_vis/bloch_t_0_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')
            
            _, ax = visualizeBloch(tdist[np.argmin(min_variance_SN_t)],t[np.argmin(min_variance_SN_t)],viewAngle=(0,0),axesLabels=True,showProjection=False);
            ax.figure.savefig('bloch_vis/bloch_t_opt_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

            _, ax = visualizeBloch(tdist[0],t[0],viewAngle=(elev,azim),axesLabels=True,showProjection=True);
            ax.figure.savefig('bloch_vis/bloch2_t_0_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

            _, ax = visualizeBloch(tdist[np.argmin(min_variance_SN_t)],t[np.argmin(min_variance_SN_t)],viewAngle=(elev,azim),axesLabels=True,showProjection=True);
            ax.figure.savefig('bloch_vis/bloch2_t_opt_{}_N_{}_{}_{}_{}.png'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range), format='png',bbox_inches = 'tight')

            filename = 'bloch_vis/bloch_animate_{}_N_{}_{}_{}_{}.gif'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range)
            animateBloch(t,tdist,sphere=[],viewAngle=(30,-45),showProjection=True,axesLabels=True,showAxes=True,saveBool=True,filename=filename,fps=4)
