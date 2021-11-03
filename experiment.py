import util
import numpy as np
from itertools import combinations

LATTICE_SPACING = 1.0

class SpinLattice(util.Graph):

    def __init__(self, lattice_X, lattice_Y, lattice_spacing, prob_full=1.0, is_triangular=False):
        super().__init__(vert_dict={}, edge_dict={})
        lattice_coords = util.get_lattice_coords(2, (lattice_X, lattice_Y), lattice_spacing, is_triangular)
        spin = 0
        for i in range(len(lattice_coords)):
            if np.random.uniform() <= prob_full: # turn off cells randomly
                self.add_vertex(spin, lattice_coords[i])
                spin += 1

    def turn_on_interactions(self, interaction_fn):
        spins = self.get_vertices()
        for spin1, spin2 in combinations(spins, 2):
            dist = util.euclidean_dist_2D(self.get_coord(spin1), self.get_coord(spin2))
            strength = interaction_fn(dist)
            if strength > 0:
                self.add_edge(spin1, spin2, strength)

    def get_interactions(self):
        return [[weight, v1, v2] for (v1, v2), weight in self.get_edge_dict().items()]

    def get_spins(self):
        return self.get_vertices()

    def get_num_spins(self):
        return self.get_num_vertices()

class FreeSpins(util.Graph):

    def __init__(self, num_particles):
        super().__init__()
        self.num_particles = num_particles
        for spin in range(self.num_particles):
            self.add_vertex(spin, (0, 0))

    def turn_on_interactions(self, interaction_fn):
        spins = self.get_vertices()
        for spin1, spin2 in combinations(spins, 2):
            strength = interaction_fn()
            if strength > 0:
                self.add_edge(spin1, spin2, strength)

def power_decay_fn(radius, alpha=6.0):
    return lambda r: 1. / (1. + np.power(r / radius, alpha))

def power_law(alpha):
    return lambda r: 1. / np.power(r, alpha)

def logistic_decay_fn(radius, beta=6.0):
    # shifts logistic decay function by radius
    return lambda r: 1.0 / (1.0 + np.exp(beta * (r - radius)))

def step_fn(radius, alpha=None):
    return lambda r: 1. * (r <= radius)

def random(radius=None, alpha=None):
    # np.random.seed(30)
    def fn(r=1):
        return np.random.uniform() * (r / r)
    return fn
    # return lambda r : np.random.uniform() * (r / r)

def none(radius=None, alpha=None):
    def fn(r=1):
        return 0 * r
    return fn
    # return lambda r: 0 * r