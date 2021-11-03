import argparse
import experiment
import numpy as np
import util

def get_interaction_fn(interaction_shape):
    if interaction_shape == 'random':
        interaction_fn = experiment.random
    elif interaction_shape == 'step_fn':
        interaction_fn = experiment.step_fn
    elif interaction_shape == 'power_decay_fn':
        interaction_fn = experiment.power_decay_fn
    elif interaction_shape == 'power_law':
        interaction_fn = experiment.power_law
    return interaction_fn

def initialize_experiment(structure, system_size, fill, interaction_shape, interaction_range):
    interaction_fn = get_interaction_fn(interaction_shape)
    if structure == 'free':
        assert isinstance(system_size, int)
        assert interaction_shape == 'random'
        assert interaction_range == 'NA'
        setup = experiment.FreeSpins(system_size)
        setup.turn_on_interactions(interaction_fn())
    else:
        assert isinstance(system_size, (list, tuple, np.ndarray))
        setup = experiment.SpinLattice(system_size[0], system_size[1], experiment.LATTICE_SPACING, prob_full=fill, is_triangular=(structure=='triangular'))
        setup.turn_on_interactions(interaction_fn(interaction_range))
    return setup

def configure(specify_range=False, specify_coupling=False, specify_trotter=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--structure", default='square', help="System structure (options: square, triangular, free)")
    parser.add_argument("-n", "--size", help="System size (options: integer N or integers L_x, L_y, depending on structure)", nargs='+', type=int)
    parser.add_argument("-f", "--fill", default=100.0, help="Fill (options: float in range [0, 100])", type=float)
    parser.add_argument("-i", "--interaction", default='power_decay_fn', help="Interaction shape (options: step_fn, power_decay_fn, random, power_law)")
    parser.add_argument("-t", "--instance", default=0, help="Instance number", type=int)
    if specify_range:
        parser.add_argument("-r", "--range", default=1., help="Interaction range", type=float)
    if specify_coupling:
        parser.add_argument("-c", "--coupling", default=1., help="J_z - J_perp (coupling), or h (transverse field strength), by context", type=float)
    if specify_trotter:
        parser.add_argument("-a", "--trotter", default=1., help="Number of troterrization steps", type=int)
    args = parser.parse_args()
    
    structure = args.structure
    
    system_size = args.size
    system_size = tuple(system_size) if len(system_size) > 1 else system_size[0]
    
    fill = args.fill / 100.0
    if fill == -1:
        fill = np.random.uniform(0.5, 1.0)
    
    interaction_shape = args.interaction
    
    if interaction_shape == 'random':
        interaction_param_name = 'N/A'
    elif interaction_shape == 'power_law':
        interaction_param_name = 'exp'
    else:
        interaction_param_name = 'radius'
    
    i = args.instance
    
    if specify_range:
        interaction_range = args.range / 2.
    else:
        if interaction_shape == 'random':
            interaction_range_list = ['N/A']
        elif interaction_shape == 'power_law':
            interaction_range_list = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        else:
            assert isinstance(system_size, (list, tuple, np.ndarray))
            interaction_range_list = np.arange(1.0, util.euclidean_dist_2D((0.0, 0.0), (system_size[0] * experiment.LATTICE_SPACING, system_size[1] * experiment.LATTICE_SPACING)), experiment.LATTICE_SPACING)
    
    if specify_coupling:
        coupling = args.coupling
    
    if specify_trotter:
        n_trotter_steps = args.trotter

    configuration = [structure, system_size, fill, interaction_shape, interaction_param_name]
    
    if specify_range:
        configuration.append(interaction_range)
    else:
        configuration.append(interaction_range_list)
    
    if specify_coupling:
        configuration.append(coupling)
    
    if specify_trotter:
        configuration.append(n_trotter_steps)
        
    configuration.append(i)
    return configuration