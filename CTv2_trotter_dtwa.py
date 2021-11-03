import numpy as np
import setup
import spin_dynamics as sd
import os
import util
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, n_trotter_steps, instance = setup.configure(specify_range=True, specify_trotter=True)
    
    method = 'CTv2'
    spin_system = sd.SpinOperators_DTWA(structure, system_size, fill)
    N = spin_system.N
    psi_0 = spin_system.get_init_state('x')
    B = spin_system.get_transverse_Hamiltonian('y')

    for interaction_range in [interaction_range]:
        for J in [1.]:
            H_1 = spin_system.get_Ising_Hamiltonian(2., interaction_range)
            H_2 = spin_system.get_Ising_Hamiltonian(1., interaction_range)
            spin_evolution = sd.SpinEvolution((H_1, H_2), psi_0, B=B)

            cont_filename = 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}'.format(method, N, 'power_law', 'exp', interaction_range, J)
            cont_dirname = '../{}_dtwa'.format(method)
            if cont_filename in os.listdir(cont_dirname):
                cont_observed_t = util.read_observed_t('{}/{}'.format(cont_dirname, cont_filename))
                cont_variance_SN_t, cont_variance_norm_t, cont_angle_t, cont_t = cont_observed_t['min_variance_SN'], cont_observed_t['min_variance_norm'], cont_observed_t['opt_angle'], cont_observed_t['t']
                
                idx_first_cont_variance_SN = argrelextrema(cont_variance_SN_t, np.less)[0][0]
                total_T = cont_t[idx_first_cont_variance_SN]

                t_it = np.linspace(1., 1., 1)
                for steps in [n_trotter_steps]:
                    params = np.ones(2 * steps) * (total_T / steps)
                    tdist, t = spin_evolution.trotter_evolve_vary(params, t_it, store_states=True, discretize_time=True)
                    meanConfig_evol = np.mean(tdist,axis=1)

                    min_variance_SN_t, min_variance_norm_t, opt_angle_t = spin_system.get_squeezing(tdist, meanConfig_evol)
                    results_t = spin_system.get_observed(tdist, meanConfig_evol)
                    results_t['min_variance_SN'] = min_variance_SN_t
                    results_t['min_variance_norm'] = min_variance_norm_t
                    results_t['opt_angle'] = opt_angle_t
                    results_t['t'] = t

                    util.store_observed_t(results_t, 'observables_vs_t_{}_N_{}_{}_{}_{}_J_{}_steps_{}'.format(method, spin_system.N, interaction_shape, interaction_param_name, interaction_range, J, steps))

                        
    # total_T_vs_N_vs_range = {}
    # total_T_vs_N_vs_range[0.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.234, 0.14700000000000002, 0.07500000000000001, 0.045000000000000005, 0.027, 0.015]))
    # total_T_vs_N_vs_range[0.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.366, 0.3, 0.222, 0.17700000000000002, 0.138, 0.09000000000000001]))
    # total_T_vs_N_vs_range[1.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.47400000000000003, 0.468, 0.441, 0.423, 0.399, 0.37500000000000006]))
    # total_T_vs_N_vs_range[1.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.522, 0.546, 0.5670000000000001, 0.5790000000000001, 0.585, 0.5850000000000001]))
    # total_T_vs_N_vs_range[2.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.531, 0.552, 0.5730000000000001, 0.588, 0.591, 0.6000000000000001]))
    # total_T_vs_N_vs_range[2.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.528, 0.543, 0.558, 0.5670000000000001, 0.5670000000000001, 0.5700000000000001]))
    # total_T_vs_N_vs_range[3.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.522, 0.534, 0.546, 0.555, 0.552, 0.555]))
    # total_T_vs_N_vs_range[3.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.516, 0.528, 0.54, 0.546, 0.543, 0.54]))
    # total_T_vs_N_vs_range[4.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.513, 0.522, 0.534, 0.54, 0.537, 0.54]))
    # total_T_vs_N_vs_range[4.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.51, 0.522, 0.531, 0.534, 0.531, 0.54]))
    # total_T_vs_N_vs_range[5.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.507, 0.519, 0.528, 0.531, 0.528, 0.54]))
    # total_T_vs_N_vs_range[5.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.507, 0.516, 0.528, 0.531, 0.528, 0.54]))
    # total_T_vs_N_vs_range[6.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [0.504, 0.516, 0.525, 0.531, 0.525, 0.525]))

    # min_variance_SN_vs_N_vs_range = {}
    # min_variance_SN_vs_N_vs_range[0.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.2643255580751745+0j), (0.15313228644981478+0j), (0.06645637577770626+0j), (0.0324331560189242+0j), (0.017792362961227135+0j), (0.02502876242185012+0j)]))
    # min_variance_SN_vs_N_vs_range[0.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.28650358797666853+0j), (0.1719298104417085+0j), (0.07971308127327577+0j), (0.04189082641076615+0j), (0.02559479572609471+0j), (0.0143154602956668+0j)]))
    # min_variance_SN_vs_N_vs_range[1.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.34236548314903054+0j), (0.23243575338764622+0j), (0.12952100824626334+0j), (0.08189521483925215+0j), (0.05819900749241233+0j), (0.03596111997718744+0j)]))
    # min_variance_SN_vs_N_vs_range[1.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.41406874237788727+0j), (0.33160761840644665+0j), (0.24199333368780807+0j), (0.1994494970849258+0j), (0.18146255073528225+0j), (0.157888485919671+0j)]))
    # min_variance_SN_vs_N_vs_range[2.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.47737681283347466+0j), (0.42499303825583495+0j), (0.36085990140380625+0j), (0.3338841803006385+0j), (0.33425625652176305+0j), (0.3184372932364217+0j)]))
    # min_variance_SN_vs_N_vs_range[2.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5233441376359026+0j), (0.4895663517594555+0j), (0.4392642397071502+0j), (0.42026502956574924+0j), (0.4289846991973973+0j), (0.41534410807409555+0j)]))
    # min_variance_SN_vs_N_vs_range[3.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5543456629270804+0j), (0.5305419089906293+0j), (0.4860703140423861+0j), (0.47035070236237325+0j), (0.48218789178747984+0j), (0.4688024991395717+0j)]))
    # min_variance_SN_vs_N_vs_range[3.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5748434062412726+0j), (0.5563941308815435+0j), (0.5144597233354304+0j), (0.5002905699965098+0j), (0.5135441365460016+0j), (0.5001821163379299+0j)]))
    # min_variance_SN_vs_N_vs_range[4.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5884076901222134+0j), (0.5729562493205971+0j), (0.5322278168220924+0j), (0.5189160184758863+0j), (0.5329269392524856+0j), (0.5193831301201627+0j)]))
    # min_variance_SN_vs_N_vs_range[4.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.5974447038777907+0j), (0.583753218504027+0j), (0.5436659459059768+0j), (0.5308689932680419+0j), (0.5453262969078542+0j), (0.5317034047487654+0j)]))
    # min_variance_SN_vs_N_vs_range[5.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6035196046707608+0j), (0.5909026938913102+0j), (0.5511837128940503+0j), (0.5387179341886948+0j), (0.5534559821659437+0j), (0.5397925143973256+0j)]))
    # min_variance_SN_vs_N_vs_range[5.5] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6076387182803387+0j), (0.5957042870831835+0j), (0.5562083306019326+0j), (0.5439515365226747+0j), (0.5588753874783245+0j), (0.5451916162385702+0j)]))
    # min_variance_SN_vs_N_vs_range[6.] = dict(zip([10, 20, 50, 100, 200, 500, 1000], [(0.6104514516147783+0j), (0.5989561122723824+0j), (0.5595995956397998+0j), (0.5474924149519805+0j), (0.562537552004784+0j), (0.5488174006297087+0j)]))

