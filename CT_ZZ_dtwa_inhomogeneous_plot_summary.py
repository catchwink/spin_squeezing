import numpy as np
import setup
import spin_dynamics as sd
import util
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    for N in [1000]:    
        
                fig = plt.figure(figsize=(8,6))
                plt.title(r'Inhomogeneous (Gaussian) distribution, Rydberg dressing, N = {}'.format(N))
                plt.xlabel(r'$t$')
                plt.ylabel(r'$N \cdot \langle {S_\alpha}^2 \rangle / {\langle S_x \rangle}^2$')
                    
                dirname = 'inhomogeneous_dtwa'
                min_squeezing = 1.
                for method in ['CT', 'CTv2', 'ZZ', 'XY']:
                    filename = 'observables_vs_t_{}_inhomogeneous_N_{}_RD_J_{}'.format(method, N, 1.)
                    if filename in os.listdir(dirname):
                            observed_t = util.read_observed_t('{}/{}'.format(dirname, filename))
                            variance_SN_t, variance_norm_t, angle_t, t = observed_t['min_variance_SN'], observed_t['min_variance_norm'], observed_t['opt_angle'], observed_t['t']
                            # color = 'r' if method == 'CT' else 'b'
                            plt.plot(t, variance_SN_t, 'o', label=method)
                            min_squeezing = min(min_squeezing, min(variance_SN_t))
                plt.ylim(bottom=0, top=1.2)
                plt.xlim(left=0., right=0.005)
                plt.legend()
                plt.tight_layout()
                plt.savefig('{}/plots/N_{}.png'.format(dirname, N))
                plt.close()