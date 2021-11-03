import numpy as np
import setup
import spin_dynamics as sd
import util
import experiment_realistic
from scipy.signal import argrelextrema

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, instance = setup.configure(specify_range=True)
    structure = 'inhomogeneous'
    fill = 'N/A'
    N = system_size[0] * system_size[1]
    interaction_shape = 'RD'
    interaction_param_name = 'N/A'
    interaction_range = 'N/A'
    print(structure, system_size, fill, interaction_shape, interaction_param_name, interaction_range, instance)
    experiment = experiment_realistic.SqueezingCalculations(interaction_type='RD', N_atoms=N, detuning=16e6, rabi_freq=1.1e6, C_6=100e9)
    with open('inhomogeneous_instance_N_{}.npy'.format(N), 'wb') as f:
        np.save(f, experiment.points)
        np.save(f, -experiment.interactionMatrix)