from collections import defaultdict
import csv
from DTWA.TamLib import getLatticeCoord
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#### SYSTEM SETUP CLASSES & HELPERS ####
class Vertex(object):

    def __init__(self, id, coord):
        self.id = id
        self.coord = coord
        self.adjacent = defaultdict(int)

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def get_id(self):
        return self.id
    
    def get_coord(self):
        return self.coord
    
    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph(object):

    def __init__(self, vert_dict={}, edge_dict={}):
        self.vert_dict = vert_dict
        self.edge_dict = edge_dict
        self.num_vertices = len(self.vert_dict)

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, id, coord):
        self.num_vertices += 1
        new_vertex = Vertex(id, coord)
        self.vert_dict[id] = new_vertex
        return new_vertex

    def add_edge(self, frm, to, weight = 0.0):
        if frm not in self.vert_dict or to not in self.vert_dict:
            raise RuntimeError('Both vertices of the edge must be present')
        self.vert_dict[frm].add_neighbor(to, weight)
        self.vert_dict[to].add_neighbor(frm, weight)
        self.edge_dict[(frm, to)] = weight

    def get_vertices(self):
        return list(self.vert_dict.keys())
    
    def get_vertex(self, v):
        if v not in self.vert_dict:
            raise RuntimeError('Vertex must be present')
        return self.vert_dict[v]

    def get_coord(self, v):
        if v not in self.vert_dict:
            raise RuntimeError('Vertex must be present')
        return self.vert_dict[v].get_coord()
    
    def get_vertex_dict(self):
        return self.vert_dict
    
    def get_edge_dict(self):
        return self.edge_dict
    
    def get_edge(self, frm, to):
        if frm not in self.vert_dict or to not in self.vert_dict:
            raise RuntimeError('Both vertices of the edge must be present')
        return self.vert_dict[frm].get_weight(to)

    def get_neighbors(self, v):
        if v not in self.vert_dict:
            raise RuntimeError('Vertex must be present')
        return self.vert_dict[v].get_connections()

    def get_num_vertices(self):
        return self.num_vertices

def euclidean_dist_2D(loc1, loc2):
    return np.linalg.norm(np.array(loc1) - np.array(loc2))

def get_lattice_coords(num_lattice_dims, lattice_dims, lattice_spacing, is_triangular=False):
    lattice_coords = getLatticeCoord(num_lattice_dims, lattice_dims, lattice_spacing)
    if is_triangular:
        lattice_coords[:, 1] += (lattice_coords[:, 0] % 2) / 2
        lattice_coords[:, 0] *= np.sqrt(3) / 2
    return lattice_coords

def store_observed_t(observed_t, filename):
    with open(filename, 'w') as outfile:
        header = observed_t.keys()
        dict_writer = csv.DictWriter(outfile, header)
        dict_writer.writeheader()
        observed_t_inverted = [{} for _ in range(len(observed_t[list(header)[0]]))]
        for obs_name, obs_vs_t in observed_t.items():
            for i, obs in enumerate(obs_vs_t):
                observed_t_inverted[i][obs_name] = obs
        dict_writer.writerows(observed_t_inverted)

def read_observed_t(filename):
    observed_t = defaultdict(list)
    with open(filename, 'r') as infile:
        dict_reader = csv.DictReader(infile)
        for row in dict_reader:
            for obs_name, obs in row.items():
                if obs_name == 't':
                    obs = float(obs)
                else:
                    obs = complex(obs)
                observed_t[obs_name].append(obs)
        for obs_name, obs_t in observed_t.items():
            observed_t[obs_name] = np.array(obs_t)
    return observed_t


### PLOTS

def plot_variance_SN_vs_t(variance_SN, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='.', J_eff=None):

    fig = plt.figure()
    title = '{}, N = {}, {}, {} = {}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else 'J_eff * t'
    plt.xlabel(xlabel)

    plt.ylabel('N * <S_a^2> / <S_x>^2')
    
    plt.ylim(bottom=0., top=1.)

    Jt = t if J_eff is None else J_eff * t
    plt.plot(Jt, variance_SN)

    filename = 'variance_SN_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_variance_norm_vs_t(variance_norm, t, method, N, interaction_shape, interaction_param_name, interaction_range, variance_norm_perp=None, dirname='.', J_eff=None):

    fig = plt.figure()
    title = '{}, N = {}, {}, {} = {}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else 'J_eff * t'
    plt.xlabel(xlabel)

    plt.ylabel('(normalized) <S_a^2>')
    
    Jt = t if J_eff is None else J_eff * t
    plt.plot(Jt, variance_norm, color='blue', label='optimal angle')
    if variance_norm_perp is not None:
        plt.plot(Jt, variance_norm_perp, color='green', label='pi/2 away from optimal angle')
    
    plt.legend()

    filename = 'variance_norm_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_angle_vs_t(angle, t, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='.', J_eff=None):

    fig = plt.figure()
    title = '{}, N = {}, {}, {} = {}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else 'J_eff * t'
    plt.xlabel(xlabel)

    plt.ylabel('squeezing angle')
    
    Jt = t if J_eff is None else J_eff * t
    plt.plot(Jt, angle)

    filename = 'angle_vs_t_{}_N_{}_{}_{}_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_variance_SN_vs_t_all_ranges(variance_SN_vs_range_vs_method, t_vs_range_vs_method, N, interaction_shape, interaction_param_name, dirname='.', J_eff=None):

    fig = plt.figure(figsize=(7.2,4.8))
    title = 'N = {}, {}'.format(N, interaction_shape)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else '|J_eff| * t'
    plt.xlabel(xlabel)

    plt.ylabel('N * <S_a^2> / <S_x>^2')

    plt.ylim(bottom=0., top=1.)

    for method in variance_SN_vs_range_vs_method:
        variance_SN_vs_range = variance_SN_vs_range_vs_method[method]
        t_vs_range = t_vs_range_vs_method[method]
        color_idx = np.linspace(1. / len(variance_SN_vs_range), 1., len(variance_SN_vs_range))
        if method == 'ZZ':
            color_scale = 'Blues'
        elif method == 'XY':
            color_scale = 'Reds'
        else:
            color_scale = 'Greens'
        for i, (range, variance_SN) in zip(color_idx, variance_SN_vs_range.items()):
            Jt = t_vs_range[range] if J_eff is None else np.abs(J_eff) * t_vs_range[range]
            plt.plot(Jt, variance_SN, color=plt.cm.get_cmap(color_scale)(i), label='{}, {} = {}'.format(method, interaction_param_name, range))
        range_list = list(variance_SN_vs_range.keys())
        norm = mpl.colors.Normalize(vmin=0, vmax=1) 
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(color_scale), norm=norm) 
        sm.set_array([])         
        colorbar = plt.colorbar(sm, ticks = color_idx, label='{} ({})'.format(interaction_param_name, method))
        colorbar.ax.set_yticklabels(range_list)   

    plt.legend()
        
    filename = 'variance_SN_vs_t_N_{}_{}_all_ranges'.format(N, interaction_shape)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))

    plt.ylim(bottom=0.01, top=10)
    plt.yscale('log')
    plt.tight_layout()

    filename = 'variance_SN_log_vs_t_N_{}_{}_all_ranges'.format(N, interaction_shape)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_variance_norm_vs_t_all_ranges(variance_norm_vs_range_vs_method, t_vs_range_vs_method, N, interaction_shape, interaction_param_name, dirname='.', J_eff=None):

    fig = plt.figure(figsize=(7.2,4.8))
    title = 'N = {}, {}'.format(N, interaction_shape)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else '|J_eff| * t'
    plt.xlabel(xlabel)

    plt.ylabel('(normalized) <S_a^2>')

    for method in variance_norm_vs_range_vs_method:
        variance_norm_vs_range = variance_norm_vs_range_vs_method[method]
        t_vs_range = t_vs_range_vs_method[method]
        color_idx = np.linspace(1. / len(variance_norm_vs_range), 1., len(variance_norm_vs_range))
        if method == 'ZZ':
            color_scale = 'Blues'
        elif method == 'XY':
            color_scale = 'Reds'
        else:
            color_scale = 'Greens'
        for i, (range, variance_norm) in zip(color_idx, variance_norm_vs_range.items()):
            Jt = t_vs_range[range] if J_eff is None else np.abs(J_eff) * t_vs_range[range]
            plt.plot(Jt, variance_norm, color=plt.cm.get_cmap(color_scale)(i), label='{}, {} = {}'.format(method, interaction_param_name, range))
        range_list = list(variance_norm_vs_range.keys())
        norm = mpl.colors.Normalize(vmin=0, vmax=1) 
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(color_scale), norm=norm) 
        sm.set_array([])         
        colorbar = plt.colorbar(sm, ticks = color_idx, label='{} ({})'.format(interaction_param_name, method))
        colorbar.ax.set_yticklabels(range_list)   
        
    plt.legend()

    filename = 'variance_norm_vs_t_N_{}_{}_all_ranges'.format(N, interaction_shape)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_min_variance_SN_vs_range(variance_SN_vs_range_vs_method, N, interaction_shape, interaction_param_name, dirname='.', J_eff=None):

    fig = plt.figure()
    title = 'N = {}, {}'.format(N, interaction_shape)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    plt.xlabel(interaction_param_name)

    plt.ylabel('minimum N * <S_a^2> / <S_x>^2')

    plt.ylim(bottom=0., top=1.)

    for method in variance_SN_vs_range_vs_method:
        variance_SN_vs_range = variance_SN_vs_range_vs_method[method]
        ranges = []
        min_variance_SN = []
        for range, variance_SN in variance_SN_vs_range.items():
            ranges.append(range)
            min_variance_SN.append(min(variance_SN[:200]))
        plt.plot(ranges, min_variance_SN, label='{}'.format(method))

        range_list = list(variance_SN_vs_range.keys())
        norm = mpl.colors.Normalize(vmin=0, vmax=1) 
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(color_scale), norm=norm) 
        sm.set_array([])         
        colorbar = plt.colorbar(sm, ticks = color_idx, label='{} ({})'.format(interaction_param_name, method))
        colorbar.ax.set_yticklabels(range_list)   
    
    plt.legend()

    filename = 'min_variance_SN_vs_range_N_{}_{}'.format(N, interaction_shape)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))

    plt.yscale('log')
    plt.ylim(bottom=0.01, top=None)

    filename = 'min_variance_SN_log_vs_range_N_{}_{}'.format(N, interaction_shape)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_variance_SN_vs_t_all_J_effs(variance_SN_vs_J_eff, t_vs_J_eff, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='.'):

    fig = plt.figure(figsize=(9,6))
    title = '{}, N = {}, {}, {} = {}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else '|J_eff| * t'
    plt.xlabel(xlabel)

    plt.ylabel('N * <S_a^2> / <S_x>^2')

    plt.ylim(bottom=0., top=1.)

    opt_solns = {}
    color_idx = np.linspace(1. / len(variance_SN_vs_J_eff), 1., len(variance_SN_vs_J_eff))
    for i, (J_eff, variance_SN) in zip(color_idx, variance_SN_vs_J_eff.items()):
        Jt = np.abs(J_eff) * t_vs_J_eff[J_eff]
        plt.plot(Jt, variance_SN, color=plt.cm.get_cmap("Reds")(i), label='J_eff = {}'.format(J_eff))
        opt_solns[J_eff] = (t_vs_J_eff[J_eff][np.argmin(variance_SN)], np.min(variance_SN))
    
    plt.legend()

    filename = 'variance_SN_vs_t_{}_N_{}_{}_{}_{}_all_J_effs'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    
    plt.yscale('log')
    plt.ylim(bottom=0.01, top=None)

    filename = 'variance_SN_log_vs_t_{}_N_{}_{}_{}_{}_all_J_effs'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

    return opt_solns

def plot_min_variance_SN_vs_J_eff_all_Ns(variance_SN_vs_J_eff_vs_N, method, interaction_shape, interaction_param_name, interaction_range, dirname='.'):

    fig = plt.figure()
    title = '{}, {}, {} = {}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.title(title)

    plt.xlabel('- J_eff = - J_z + J_perp')

    plt.ylabel('minimum N * <S_a^2> / <S_x>^2')

    plt.ylim(bottom=0., top=1.)

    color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_N), 1., len(variance_SN_vs_J_eff_vs_N))
    min_variance_SN_vs_N = {}
    for i, (N, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_N.items()):
        J_effs = []
        min_variance_SN = []
        for J_eff, variance_SN in sorted(variance_SN_vs_J_eff.items()):
            if J_eff not in [1., -1.]:
                J_effs.append(J_eff)
                min_variance_SN.append(min(variance_SN))
            else:
                color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                linestyle = 'dashed' if J_eff == 1. else 'solid'
                plt.hlines(min(min_variance_SN), 0, 0.4, color=color, linestyle='dashed', label='N = {}, J_z - J_perp = {}'.format(N, J_eff))
        plt.plot(- np.array(J_effs), min_variance_SN, marker='.', color=plt.cm.get_cmap("Reds")(i), label='N = {}'.format(N))
        min_variance_SN_vs_N[N] = min(min_variance_SN)
 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

    plt.tight_layout()

    filename = 'min_variance_SN_vs_J_eff_{}_{}_{}_{}_all_Ns'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))

    plt.xscale('log')
    plt.tight_layout()

    filename = 'min_variance_SN_vs_log_J_eff_{}_{}_{}_{}_all_Ns'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    
    plt.yscale('log')
    plt.ylim(bottom=0.01, top=None)
    plt.tight_layout()

    filename = 'min_variance_SN_log_vs_J_eff_{}_{}_{}_{}_all_Ns'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

    return min_variance_SN_vs_N

def plot_min_variance_SN_vs_J_eff_all_ranges(variance_SN_vs_J_eff_vs_range, method, N, interaction_shape, interaction_param_name, dirname='.'):

    fig = plt.figure()
    title = '{}, N = {}, {}'.format(method, N, interaction_shape)
    plt.title(title)

    plt.xlabel('- J_eff = - J_z + J_perp')

    plt.ylabel('minimum N * <S_a^2> / <S_x>^2')

    plt.ylim(bottom=0., top=1.)

    color_idx = np.linspace(1. / len(variance_SN_vs_J_eff_vs_range), 1., len(variance_SN_vs_J_eff_vs_range))
    min_variance_SN_vs_range = {}
    for i, (range, variance_SN_vs_J_eff) in zip(color_idx, variance_SN_vs_J_eff_vs_range.items()):
        J_effs = []
        min_variance_SN = []
        for J_eff, variance_SN in sorted(variance_SN_vs_J_eff.items()):
            if J_eff not in [1., -1.]:
                J_effs.append(J_eff)
                min_variance_SN.append(min(variance_SN))
            else:
                color = plt.cm.get_cmap("Blues")(i) if J_eff == 1. else plt.cm.get_cmap("Greens")(i)
                linestyle = 'dashed' if J_eff == 1. else 'solid'
                plt.hlines(min(min_variance_SN), 0, 0.4, color=color, linestyle='dashed', label='{} = {}, J_z - J_perp = {}'.format(interaction_param_name, interaction_range, J_eff))
        plt.plot(- np.array(J_effs), min_variance_SN, marker='.', color=plt.cm.get_cmap("Reds")(i), label='{} = {}'.format(interaction_param_name, interaction_range))
        min_variance_SN_vs_range[range] = min(min_variance_SN)
 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

    plt.tight_layout()

    filename = 'min_variance_SN_vs_J_eff_{}_N_{}_{}_all_ranges'.format(method, N, interaction_shape)
    plt.savefig('{}/{}.png'.format(dirname, filename))

    plt.xscale('log')
    plt.tight_layout()

    filename = 'min_variance_SN_vs_log_J_eff_{}_N_{}_{}_all_ranges'.format(method, N, interaction_shape)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    
    plt.yscale('log')
    plt.ylim(bottom=0.01, top=None)
    plt.tight_layout()

    filename = 'min_variance_SN_log_vs_log_J_eff_{}_N_{}_{}_all_ranges'.format(method, N, interaction_shape)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

    return min_variance_SN_vs_N

def plot_t_opt_vs_J_eff_all_Ns(opt_solns_vs_N, method, interaction_shape, interaction_param_name, interaction_range, dirname='.'):

    def func(J_eff_inv, a):
        return a * J_eff_inv
    from scipy.optimize import curve_fit

    fig = plt.figure()
    title = '{}, {}, {} = {}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.title(title)

    plt.xlabel('1 / |J_eff|')

    plt.ylabel('t_opt')

    color_idx = np.linspace(1. / len(opt_solns_vs_N), 1., len(opt_solns_vs_N))
    for i, (N, opt_solns) in zip(color_idx, opt_solns_vs_N.items()):
        J_effs = []
        t_opt_list = []
        for J_eff, (t_opt, squeezing_opt) in opt_solns.items():
            J_effs.append(J_eff)
            t_opt_list.append(t_opt)
        popt, pcov = curve_fit(func, (1. / np.abs(J_effs)), t_opt_list) 
        plt.plot(1. / np.abs(J_effs), t_opt_list, 'o', label='N = {}'.format(N), color=plt.cm.get_cmap("Reds")(i))
        plt.plot(1. / np.abs(J_effs), func(1. / np.abs(J_effs), *popt), label='%5.3f * 1/|J_eff|' % tuple(popt), linestyle='dashed', color=plt.cm.get_cmap("Reds")(i))
    
    plt.legend()
    plt.tight_layout()
    
    filename = 't_opt_vs_inv_J_eff_{}_{}_{}_{}_all_Ns'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

    fig = plt.figure()
    title = '{}, {}, {} = {}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.title(title)

    plt.xlabel('|J_eff|')

    plt.ylabel('t_opt')

    color_idx = np.linspace(1. / len(opt_solns_vs_N), 1., len(opt_solns_vs_N))
    for i, (N, opt_solns) in zip(color_idx, opt_solns_vs_N.items()):
        J_effs = []
        t_opt_list = []
        for J_eff, (t_opt, squeezing_opt) in opt_solns.items():
            J_effs.append(J_eff)
            t_opt_list.append(t_opt)
        popt, pcov = curve_fit(func, (1. / np.abs(J_effs)), t_opt_list) 
        plt.plot(np.abs(J_effs), t_opt_list, 'o', label='N = {}'.format(N), color=plt.cm.get_cmap("Reds")(i))
        plt.plot(np.abs(J_effs), func(1. / np.abs(J_effs), *popt), label='%5.3f * 1/|J_eff|' % tuple(popt), linestyle='dashed', color=plt.cm.get_cmap("Reds")(i))
    
    plt.xscale('log')
    plt.yscale('log')

    plt.legend()
    plt.tight_layout()
    
    filename = 't_opt_vs_J_eff_{}_{}_{}_{}_all_Ns'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()


def plot_J_eff_plateau_vs_N(plateau_J_effs, Ns, method, interaction_shape, interaction_param_name, interaction_range, dirname='.'):

    def func(N, a, b, c):
        return a * (N ** (-b)) + c

    fig = plt.figure()
    title = '{}, {}, {} = {}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.title(title)

    plt.xlabel('N')

    plt.ylabel('|J_eff|')
    
    popt, pcov = curve_fit(func, Ns, np.abs(plateau_J_effs))
    plt.plot(Ns, np.abs(plateau_J_effs), 'o', label='{}, {} = {}'.format(interaction_shape, interaction_param_name, interaction_range))
    plt.plot(Ns, func(Ns, *popt), label='%5.3f / (N ^ %5.3f) + %5.3f' % tuple(popt), linestyle='dashed')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.legend()
    plt.tight_layout()

    filename = 'J_eff_plateau_vs_N_{}_{}_{}_{}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()


def plot_min_variance_SN_vs_N(min_variance_SNs_vs_J_eff, Ns_vs_J_eff, method, interaction_shape, interaction_param_name, interaction_range, dirname='.'):

    def func(N, a, b, c):
        return a * (N ** (-b)) + c

    fig = plt.figure()
    title = '{}, {}, {} = {}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.title(title)

    plt.xlabel('N')

    plt.ylabel('N * <S_a^2> / <S_x>^2')
    
    for J_eff, min_variance_SNs in min_variance_SNs_vs_J_eff.items():
        Ns = Ns_vs_J_eff
        plt.plot(Ns, min_variance_SNs, 'o', label='J_eff = {}'.format(J_eff))
        popt, pcov = curve_fit(func, Ns, min_squeezing_SN_list)
        plt.plot(Ns, func(Ns, *popt), label='%5.3f / (N ^ %5.3f) + %5.3f' % tuple(popt), linestyle='dashed')


    popt, pcov = curve_fit(func, Ns, np.abs(plateau_J_effs))
    plt.plot(Ns, np.abs(plateau_J_effs), 'o', label='{}, {} = {}'.format(interaction_shape, interaction_param_name, interaction_range))
    plt.plot(Ns, func(Ns, *popt), label='%5.3f / (N ^ %5.3f) + %5.3f' % tuple(popt), linestyle='dashed')
    
    plt.xscale('log')
    plt.ylim(bottom=0)
    
    plt.legend()
    plt.tight_layout()

    filename = 'min_variance_SN_vs_N_{}_{}_{}_{}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))

    plt.yscale('log')
    plt.ylim(bottom=None)
    plt.tight_layout()

    filename = 'min_variance_SN_log_vs_N_{}_{}_{}_{}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    plt.savefig('{}/{}.png'.format(dirname, filename))

    plt.close()


def plot_variance_SN_vs_t_trotter(variance_SN_vs_steps, t_vs_steps, total_T, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='.', J_eff=None):

    fig = plt.figure()
    title = '{}, N = {}, {}, {} = {}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else '|J_eff| * t'
    plt.xlabel(xlabel)

    plt.ylabel('N * <S_a^2> / <S_x>^2')
    
    plt.ylim(bottom=0., top=1.)

    color_idx = np.linspace(1. / len(variance_SN_vs_steps), 1., len(variance_SN_vs_steps))
    for i, (steps, variance_SN) in zip(color_idx, variance_SN_vs_steps.items()):
        Jt = t_vs_steps[steps] if J_eff is None else np.abs(J_eff) * t_vs_steps[steps]
        plt.plot(Jt, variance_SN, color=plt.cm.get_cmap("Reds")(i), label='# steps = {}, step size = {}'.format(steps, round(total_T / steps, 2)))

    plt.legend()

    filename = 'variance_SN_vs_t_trotter_{}_N_{}_{}_{}_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_variance_norm_vs_t_trotter(variance_norm_vs_steps, t_vs_steps, total_T, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='.', J_eff=None):

    fig = plt.figure()
    title = '{}, N = {}, {}, {} = {}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else '|J_eff| * t'
    plt.xlabel(xlabel)

    plt.ylabel('(normalized) <S_a^2>')
    
    plt.ylim(bottom=0., top=1.)

    color_idx = np.linspace(1. / len(variance_norm_vs_steps), 1., len(variance_norm_vs_steps))
    for i, (steps, variance_norm) in zip(color_idx, variance_norm_vs_steps.items()):
        Jt = t_vs_steps[steps] if J_eff is None else np.abs(J_eff) * t_vs_steps[steps]
        plt.plot(Jt, variance_norm, color=plt.cm.get_cmap("Reds")(i), label='# steps = {}, step size = {}'.format(steps, round(total_T / steps, 2)))

    plt.legend()

    filename = 'variance_norm_vs_t_trotter_{}_N_{}_{}_{}_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_components_vs_t_trotter(components_vs_steps, t_vs_steps, total_T, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='.', J_eff=None):

    fig = plt.figure()
    title = '{}, N = {}, {}, {} = {}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else '|J_eff| * t'
    plt.xlabel(xlabel)

    plt.ylabel('<O>')
    
    plt.ylim(bottom=0., top=1.)

    color_idx = np.linspace(1. / len(components_vs_steps), 1., len(components_vs_steps))
    for i, (steps, S) in zip(color_idx, components_vs_steps.items()):
        S_x, S_y, S_z = S
        Jt = t_vs_steps[steps] if J_eff is None else np.abs(J_eff) * t_vs_steps[steps]
        plt.plot(Jt, S_x, color=plt.cm.get_cmap("Reds")(i), label='<S_x>, # steps = {}, step size = {}'.format(steps, round(total_T / steps, 2)))
        plt.plot(Jt, S_y, color=plt.cm.get_cmap("Blues")(i), label='<S_y>, # steps = {}, step size = {}'.format(steps, round(total_T / steps, 2)))
        plt.plot(Jt, S_z, color=plt.cm.get_cmap("Greens")(i), label='<S_z>, # steps = {}, step size = {}'.format(steps, round(total_T / steps, 2)))

    plt.legend()

    filename = 'components_vs_t_trotter_{}_N_{}_{}_{}_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_signal_noise_vs_t_trotter(components_vs_steps, variance_norm_vs_steps, t_vs_steps, total_T, method, N, interaction_shape, interaction_param_name, interaction_range, dirname='.', J_eff=None):

    fig = plt.figure()
    title = '{}, N = {}, {}, {} = {}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    xlabel = 'J * t' if J_eff is None else '|J_eff| * t'
    plt.xlabel(xlabel)

    plt.ylabel('<O>')
    
    plt.ylim(bottom=0., top=1.)

    color_idx = np.linspace(1. / len(components_vs_steps), 1., len(components_vs_steps))
    for i, (steps, S) in zip(color_idx, components_vs_steps.items()):
        S_x, S_y, S_z = S
        variance_norm = variance_norm_vs_steps[steps]
        Jt = t_vs_steps[steps] if J_eff is None else np.abs(J_eff) * t_vs_steps[steps]
        plt.plot(Jt, S_x, color=plt.cm.get_cmap("Reds")(i), label='<S_x>, # steps = {}, step size = {}'.format(steps, round(total_T / steps, 2)))
        plt.plot(Jt, variance_norm, color=plt.cm.get_cmap("Blues")(i), label='(normalized) <S_a^2>, # steps = {}, step size = {}'.format(steps, round(total_T / steps, 2)))

    plt.legend()

    filename = 'components_vs_t_trotter_{}_N_{}_{}_{}_{}'.format(method, N, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()

def plot_variance_SN_vs_delta_t_trotter(variance_SN_vs_delta_t_vs_N, method, interaction_shape, interaction_param_name, interaction_range, dirname='.', J_eff=None, norm=False):

    fig = plt.figure()
    title = '{}, {}, {} = {}'.format(method, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        title += ', J_eff = {}'.format(J_eff)
    plt.title(title)

    plt.xlabel('|J_eff| * Δt')
    
    plt.ylabel('N * <S_a^2> / <S_x>^2 at t_opt of {} squeezing'.format(method))

    for N, variance_SN_vs_delta_t in variance_SN_vs_delta_t_vs_N.items():
        J_eff_delta_t = []
        variance_SN_delta_t = []
        for (J_eff, delta_t), variance_SN in variance_SN_vs_delta_t.items():
            Jt = delta_t if J_eff is None else np.abs(J_eff) * delta_t
            J_eff_delta_t.append(Jt)
            print(J_eff, delta_t, Jt)
            variance_SN_delta_t.append(variance_SN)
            if delta_t == 0:
                norm = variance_SN
        if norm:
            variance_SN_delta_t = np.array(variance_SN_delta_t) / norm
        plt.plot(J_eff_delta_t, variance_SN_delta_t, marker='o', label='N = {}'.format(N))
    
    plt.legend()
    plt.ylim(bottom=0, top=10)
    
    filename = 'variance_SN_vs_delta_t_{}_{}_{}_{}_all_Ns'.format(method, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    if norm:
        filename += '_norm'
    plt.savefig('{}/{}.png'.format(dirname, filename))
    
    plt.yscale('log')
    plt.ylim(bottom=10**(-3), top=100)
    plt.tight_layout()
    
    filename = 'variance_SN_log_vs_delta_t_{}_{}_{}_{}_all_Ns'.format(method, interaction_shape, interaction_param_name, interaction_range)
    if J_eff is not None:
        filename += '_J_eff_{}'.format(J_eff)
    plt.savefig('{}/{}.png'.format(dirname, filename))
    plt.close()


