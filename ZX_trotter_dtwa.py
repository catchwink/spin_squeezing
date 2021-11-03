import numpy as np
import setup
import spin_dynamics as sd
import os
import util
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, n_trotter_steps, instance = setup.configure(specify_range=True, specify_trotter=True)
    
    method = 'ZX'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    N = spin_system.N
    psi_0 = spin_system.get_init_state('x')
    B = spin_system.get_transverse_Hamiltonian('y')

    # for interaction_range in interaction_range_list:
    for interaction_range in [interaction_range]:

        H = spin_system.get_Ising_Hamiltonian(1., interaction_range)
        spin_evolution = sd.SpinEvolution(H, psi_0, B=B)

        t_it = np.linspace(1., 1., 1)
        for steps in [n_trotter_steps]:
            cont_filename = 'observables_vs_t_{}_N_{}_{}_{}_{}'.format('XY', N, 'power_law', 'exp', interaction_range)
            cont_dirname = '../{}_dtwa'.format('XY')
            if cont_filename in os.listdir(cont_dirname):
                cont_observed_t = util.read_observed_t('{}/{}'.format(cont_dirname, cont_filename))
                cont_variance_SN_t, cont_variance_norm_t, cont_angle_t, cont_t = cont_observed_t['min_variance_SN'], cont_observed_t['min_variance_norm'], cont_observed_t['opt_angle'], cont_observed_t['t']
                
                idx_first_cont_variance_SN = argrelextrema(cont_variance_SN_t, np.less)[0][0]
                total_T = cont_t[idx_first_cont_variance_SN]

                params = np.ones(2 * steps) * (total_T / steps)
                tdist, t = spin_evolution.trotter_evolve(params, t_it, store_states=True, discretize_time=True)
                meanConfig_evol = np.mean(tdist,axis=1)

                min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
                results_t = spin_system.get_observed(tdist, meanConfig_evol)
                results_t['min_variance_SN'] = min_variance_SN_t
                results_t['min_variance_norm'] = min_variance_norm_t
                results_t['opt_angle'] = opt_angle_t
                results_t['t'] = t

                util.store_observed_t(results_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, steps))


    # total_T_vs_N_vs_range = {}
    # total_T_vs_N_vs_range[0.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.435, 0.28500000000000003, 0.159, 0.10200000000000001, 0.066, 0.03, 0.03]))
    # total_T_vs_N_vs_range[0.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.714, 0.585, 0.501, 0.429, 0.384, 0.31500000000000006, 0.28500000000000003]))
    # total_T_vs_N_vs_range[1.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.002, 1.026, 1.158, 1.2149999999999999, 1.2839999999999998, 1.5, 1.5]))
    # total_T_vs_N_vs_range[1.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2149999999999999, 1.3439999999999999, 1.5, 1.5, 1.5, 1.5, 1.5]))
    # total_T_vs_N_vs_range[2.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.251, 1.38, 1.494, 1.5, 1.5, 1.5, 1.5]))
    # total_T_vs_N_vs_range[2.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.23, 1.3499999999999999, 1.407, 1.446, 1.4609999999999999, 1.455, 1.47]))
    # total_T_vs_N_vs_range[3.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2089999999999999, 1.275, 1.341, 1.3679999999999999, 1.392, 1.395, 1.395]))
    # total_T_vs_N_vs_range[3.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2029999999999998, 1.272, 1.3079999999999998, 1.3379999999999999, 1.329, 1.335, 1.335]))
    # total_T_vs_N_vs_range[4.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2, 1.218, 1.293, 1.2959999999999998, 1.293, 1.29, 1.305]))
    # total_T_vs_N_vs_range[4.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.2149999999999999, 1.2209999999999999, 1.269, 1.299, 1.293, 1.29, 1.29]))
    # total_T_vs_N_vs_range[5.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.146, 1.212, 1.2389999999999999, 1.26, 1.2839999999999998, 1.275, 1.275]))
    # total_T_vs_N_vs_range[5.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.1669999999999998, 1.2209999999999999, 1.242, 1.257, 1.272, 1.26, 1.26]))
    # total_T_vs_N_vs_range[6.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [1.1789999999999998, 1.2209999999999999, 1.242, 1.254, 1.2839999999999998, 1.275, 1.275]))

    # min_variance_SN_vs_N_vs_range = {}
    # min_variance_SN_vs_N_vs_range[0.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.3187557522594195+0j), (0.200022218418662+0j), (0.10387926548759195+0j), (0.061005970485207894+0j), (0.03694406317866285+0j), (0.021780251594424052+0j), (0.015858198602035235+0j)]))
    # min_variance_SN_vs_N_vs_range[0.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.32012726265203534+0j), (0.2030538881240366+0j), (0.10094572494225816+0j), (0.06221334798618659+0j), (0.0364497050343903+0j), (0.01936909544639078+0j), (0.011771853564905451+0j)]))
    # min_variance_SN_vs_N_vs_range[1.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.357197774179694+0j), (0.2241117260402238+0j), (0.10626136720553626+0j), (0.0635088101946351+0j), (0.04184846902552937+0j), (0.02102479698899758+0j), (0.013495801267634496+0j)]))
    # min_variance_SN_vs_N_vs_range[1.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.3933944660425496+0j), (0.2900270280232179+0j), (0.17937017653234508+0j), (0.13530227426892408+0j), (0.10982031905631748+0j), (0.08598025884709004+0j), (0.07966650086971928+0j)]))
    # min_variance_SN_vs_N_vs_range[2.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.4686631231913616+0j), (0.37997612366298505+0j), (0.31148836017962556+0j), (0.29074749834115077+0j), (0.26510525835127846+0j), (0.25680746034360835+0j), (0.25076795905376076+0j)]))
    # min_variance_SN_vs_N_vs_range[2.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5112363951692522+0j), (0.45885330972233723+0j), (0.40556667824049386+0j), (0.39569616698400506+0j), (0.3941256769047173+0j), (0.3762092429390486+0j), (0.3794000900364605+0j)]))
    # min_variance_SN_vs_N_vs_range[3.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5557692684073696+0j), (0.5052593980741875+0j), (0.4699810827123452+0j), (0.4635518388236521+0j), (0.45952146580324327+0j), (0.4664608019792977+0j), (0.45285551274802716+0j)]))
    # min_variance_SN_vs_N_vs_range[3.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5670255492221451+0j), (0.5278939554603594+0j), (0.5097251298257347+0j), (0.5007206937682704+0j), (0.49589922625064564+0j), (0.5000080882782595+0j), (0.49524671809143817+0j)]))
    # min_variance_SN_vs_N_vs_range[4.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5833091311772926+0j), (0.5455515351479246+0j), (0.5422806540405265+0j), (0.516929782738086+0j), (0.5219163884523763+0j), (0.5245692207638514+0j), (0.506836652495189+0j)]))
    # min_variance_SN_vs_N_vs_range[4.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5802179929807516+0j), (0.5642351535321916+0j), (0.5549683350644242+0j), (0.5478956273715833+0j), (0.5273184938235134+0j), (0.5298573107897742+0j), (0.5411648684944769+0j)]))
    # min_variance_SN_vs_N_vs_range[5.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6078529093690566+0j), (0.5782169190664381+0j), (0.5486437548300399+0j), (0.5507535087926309+0j), (0.5317498472660189+0j), (0.5465503075020792+0j), (0.5420547699510118+0j)]))
    # min_variance_SN_vs_N_vs_range[5.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6161220525169849+0j), (0.5840808588220184+0j), (0.5594419743283089+0j), (0.5555948987375099+0j), (0.5548465331215789+0j), (0.5406403473746832+0j), (0.5676516028848497+0j)]))
    # min_variance_SN_vs_N_vs_range[6.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6190377723445925+0j), (0.5975470912668818+0j), (0.5537568966317522+0j), (0.5702852666361092+0j), (0.5649508158185218+0j), (0.5497115711221022+0j), (0.5557429433142047+0j)]))

