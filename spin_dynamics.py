from DTWA import DTWA_Lib as dtwa
from DTWA import TamLib as dtwa_util
import numpy as np
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian, quantum_operator
from quspin.tools.measurements import obs_vs_time
import setup
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import diags

def op_evolve(psi, op, t):
    # Sparse matrix exponentiation
    if isinstance(op, sparse.spmatrix):
        psi_it = []
        dts = [t[0]]
        for i in range(1, len(t)):
            dts.append(t[i] - t[i - 1])
        for dt in dts:
            psi = expm_multiply(-1j*op*dt, psi)
            psi_it.append(psi)
        psi_it = np.array(psi_it)
    # QuSpin
    elif isinstance(op, hamiltonian):
        psi_it = op.evolve(psi, 0, t)
        psi_it = np.transpose(psi_it)
    # DTWA
    elif callable(op):
        psi_it = op(psi, t)
    return psi_it

def op_expectation(psi_it, op, t=None, op_name=None):
    if isinstance(op, sparse.spmatrix):
        return np.array([(np.dot(np.conjugate(psi), op @ psi) / np.dot(np.conjugate(psi), psi)) for psi in psi_it])
    elif isinstance(op, hamiltonian):
        return obs_vs_time(np.transpose(psi_it), t, {op_name: op})[op_name]

def store_observables(observables, observed_t, psi_it, t_it=None):
    observed_it = {}
    for obs_name, obs in observables.items():
        observed_it[obs_name] = op_expectation(psi_it, obs, t=t_it, op_name=obs_name)
        if obs_name not in observed_t:
            observed_t[obs_name] = observed_it[obs_name]
        else:
            observed_t[obs_name] = np.concatenate((observed_t[obs_name], observed_it[obs_name]))

def normalize(vec):
    return vec / np.linalg.norm(vec)

class SpinEvolution(object):

    def __init__(self, H, psi_0, B=None):
        self.H = H
        self.B = B
        self.psi_0 = psi_0

    def evolve(self, angles, t_it, observables=None, store_states=False):
        psi = self.psi_0
        if store_states: 
            psi_t = np.array([psi])
        if observables is not None:
            observed_t = {}
            store_observables(observables, observed_t, [psi], t_it=[0])
        t = 0
        tvec = [t]
        for it in range(len(angles)):
            t_op = angles[it] * t_it
            psi_it = op_evolve(psi, self.H, t_op)
            psi = psi_it[-1]
            if observables is not None:
                store_observables(observables, observed_t, psi_it, t_it=t_op)
            if store_states:
                psi_t = np.concatenate((psi_t, psi_it))
            tvec = np.concatenate((tvec, t_op + t))
        t += t_op[-1]
        return (psi_t, observed_t, tvec) if (store_states and (observables is not None)) else (psi_t, tvec) if store_states else (observed_t, tvec) if (observables is not None) else tvec

    def trotter_evolve(self, angles, t_it, observables=None, store_states=False, discretize_time=True):
        # S_y (instantanteous +/- pi/2 rotation) will not count towards evolution
        psi = self.psi_0
        if store_states: 
            psi_t = np.array([psi])
        if observables is not None:
            observed_t = {}
            store_observables(observables, observed_t, [psi], t_it=[0])
        t = 0
        tvec = [t]
        for it in range(len(angles)):
            t_op = angles[it] * t_it
            psi_it = op_evolve(psi, self.H, t_op)
            psi_temp = psi_it[-1]
            # Rotation about y axis (Sy)
            if it % 2 == 0:
                # rotate about y axis by +pi/2
                psi = op_evolve(psi_temp, self.B, np.pi/2 * t_it)[-1]
            else:
                # rotate about y axis by -pi/2
                psi = op_evolve(psi_temp, self.B, -np.pi/2 * t_it)[-1]
            psi_it[-1] = psi

            if not discretize_time:
                if store_states:
                    psi_t = np.concatenate((psi_t, psi_it))
                if observables is not None:
                    store_observables(observables, observed_t, psi_it, t_it=t_op)
                tvec = np.concatenate((tvec, t_op + t))
            elif it % 2 == 1:
                if store_states:
                    psi_t = np.concatenate((psi_t, [psi]))
                if observables is not None:
                    store_observables(observables, observed_t, [psi], t_it=t_op[-1:])
                tvec = np.concatenate((tvec, t_op[-1:] + t))
            t += t_op[-1]
        return (psi_t, observed_t, tvec) if (store_states and (observables is not None)) else (psi_t, tvec) if store_states else (observed_t, tvec) if (observables is not None) else tvec

    def trotter_evolve_vary(self, angles, t_it, observables=None, store_states=False, discretize_time=True):
        # S_y (instantanteous +/- pi/2 rotation) will not count towards evolution
        H_1, H_2 = self.H 
        psi = self.psi_0
        if store_states: 
            psi_t = np.array([psi])
        if observables is not None:
            observed_t = {}
            store_observables(observables, observed_t, [psi], t_it=[0])             
        t = 0
        tvec = [t]
        for it in range(len(angles)):
            t_op = angles[it] * t_it

            if it % 2 == 0:
                psi_it = op_evolve(psi, H_1, t_op)
            else:
                psi_it = op_evolve(psi, H_2, t_op)
            psi_temp = psi_it[-1]
            # Rotation about y axis (Sy)
            if it % 2 == 0:
                # rotate about y axis by +pi/2
                psi = op_evolve(psi_temp, self.B, np.pi/2 * t_it)[-1]
            else:
                # rotate about y axis by -pi/2
                psi = op_evolve(psi_temp, self.B, -np.pi/2 * t_it)[-1]
            psi_it[-1] = psi

            if not discretize_time:
                if store_states:
                    psi_t = np.concatenate((psi_t, psi_it))
                if observables is not None:
                    store_observables(observables, observed_t, psi_it, t_it=t_op)
                tvec = np.concatenate((tvec, t_op + t))
            elif it % 2 == 1:
                if store_states:
                    psi_t = np.concatenate((psi_t, [psi]))
                if observables is not None:
                    store_observables(observables, observed_t, [psi], t_it=t_op[-1:])
                tvec = np.concatenate((tvec, t_op[-1:] + t))
            t += t_op[-1]
        return (psi_t, observed_t, tvec) if (store_states and (observables is not None)) else (psi_t, tvec) if store_states else (observed_t, tvec) if (observables is not None) else tvec
    
    def trotter_evolve_twice(self, angles, t_it, observables=None, store_states=False, discretize_time=True):
        # S_x and S_y (instantanteous +/- pi/2 rotation) will not count towards evolution
        H_z, H_perp = self.H 
        B_x, B_y = self.B
        psi = self.psi_0
        if store_states: 
            psi_t = np.array([psi])
        if observables is not None:
            observed_t = {}
            store_observables(observables, observed_t, [psi], t_it=[0])             
        t = 0
        tvec = [t]
        for it in range(len(angles)):
            t_op = angles[it] * t_it
            if it % 3 == 0:
                psi_it = op_evolve(psi, H_z, t_op)
            else:
                psi_it = op_evolve(psi, H_perp, t_op)
            psi_temp = psi_it[-1]
            # Rotation about x axis (Sx)
            if it % 3 == 0:
                # rotate about x axis by +pi/2
                psi = op_evolve(psi_temp, B_x, np.pi/2 * t_it)[-1]
            elif it % 3 == 1:
                # rotate about x axis by -pi/2
                psi = op_evolve(psi_temp, B_x, -np.pi/2 * t_it)[-1]
            # Rotation about y axis (Sy)
                # rotate about y axis by +pi/2
                psi = op_evolve(psi, B_y, np.pi/2 * t_it)[-1]
            else:
                # rotate about y axis by -pi/2
                psi = op_evolve(psi_temp, B_y, -np.pi/2 * t_it)[-1]
            psi_it[-1] = psi

            if not discretize_time:
                if store_states:
                    psi_t = np.concatenate((psi_t, psi_it))
                if observables is not None:
                    store_observables(observables, observed_t, psi_it, t_it=t_op)
                tvec = np.concatenate((tvec, t_op + t))
            elif it % 3 == 2:
                if store_states:
                    psi_t = np.concatenate((psi_t, [psi]))
                if observables is not None:
                    store_observables(observables, observed_t, [psi], t_it=t_op[-1:])
                tvec = np.concatenate((tvec, t_op[-1:] + t))
            t += t_op[-1]
        return (psi_t, observed_t, tvec) if (store_states and (observables is not None)) else (psi_t, tvec) if store_states else (observed_t, tvec) if (observables is not None) else tvec

    def trotter_evolve_direct(self, angles, t_it, observables=None, store_states=False, discretize_time=True):
        psi = self.psi_0
        if store_states: 
            psi_t = np.array([psi])
        if observables is not None:
            observed_t = {}
            store_observables(observables, observed_t, [psi], t_it=[0])
        t = 0
        tvec = [t]
        for it in range(len(angles)):
            t_op = angles[it] * t_it

            if it % 2 == 0:
                psi_it = op_evolve(psi, self.H, t_op)
            else:
                psi_it = op_evolve(psi, self.B, t_op)
            psi = psi_it[-1]

            if not discretize_time:
                if store_states:
                    psi_t = np.concatenate((psi_t, psi_it))
                if observables is not None:
                    store_observables(observables, observed_t, psi_it, t_it=t_op)
                tvec = np.concatenate((tvec, t_op + t))
            elif it % 2 == 1:
                if store_states:
                    psi_t = np.concatenate((psi_t, [psi]))
                if observables is not None:
                    store_observables(observables, observed_t, [psi], t_it=t_op[-1:])
                tvec = np.concatenate((tvec, t_op[-1:] + t))
            t += t_op[-1]
        return (psi_t, observed_t, tvec) if (store_states and (observables is not None)) else (psi_t, tvec) if store_states else (observed_t, tvec) if (observables is not None) else tvec

class SpinOperators_QuSpin(object):

    def __init__(self, structure, system_size, fill, interaction_shape, interaction_range):
        self.N = 0
        while self.N <= 0:
            experiment = setup.initialize_experiment(structure, system_size, fill, interaction_shape, interaction_range)
            self.N = experiment.get_num_spins()
        
        # Spin-1/2 basis
        self.basis = spin_basis_general(self.N, S='1/2', pauli=0)
        # Ising interaction terms for interacting spins (no double-counting)
        self.J = experiment.get_interactions()
        # Transverse field terms for all spins
        self.h = [[1, i] for i in range(self.N)]
        # Identity for all pairs of spins (double-counting)
        self.p = [[1.0, v1, v2] for v1 in experiment.get_spins() for v2 in experiment.get_spins()]
        # Identity for all individual spins
        self.s = [[1, i] for i in range(self.N)]

    def get_observables(self):
        S_x = hamiltonian([["x", self.s]], [], basis=self.basis, check_herm=False, check_symm=False)
        S_y = hamiltonian([["y", self.s]], [], basis=self.basis, check_herm=False, check_symm=False)
        S_z = hamiltonian([["z", self.s]], [], basis=self.basis, check_herm=False, check_symm=False)
        S_plus = hamiltonian([["+", self.s]], [], basis=self.basis, check_herm=False, check_symm=False)
        S_minus = hamiltonian([["-", self.s]], [], basis=self.basis, check_herm=False, check_symm=False)
        S_plus_S_z = hamiltonian([["+z", self.p]], [], basis=self.basis, check_herm=False, check_symm=False)
        S_minus_S_z = hamiltonian([["-z", self.p]], [], basis=self.basis, check_herm=False, check_symm=False)
        Sx_sq = hamiltonian([["xx", self.p]], [], basis=self.basis, check_herm=False, check_symm=False)
        Sy_sq = hamiltonian([["yy", self.p]], [], basis=self.basis, check_herm=False, check_symm=False)
        Sz_sq = hamiltonian([["zz", self.p]], [], basis=self.basis, check_herm=False, check_symm=False)
        Sy_Sz = hamiltonian([["yz", self.p]], [], basis=self.basis, check_herm=False, check_symm=False)
        Sz_Sy = hamiltonian([["zy", self.p]], [], basis=self.basis, check_herm=False, check_symm=False)
        
        observables = dict(S_x=S_x, S_y=S_y, S_z=S_z, S_plus=S_plus, S_minus=S_minus, S_plus_S_z=S_plus_S_z, S_minus_S_z=S_minus_S_z, Sx_sq=Sx_sq, Sy_sq=Sy_sq, Sz_sq=Sz_sq, Sy_Sz=Sy_Sz, Sz_Sy=Sz_Sy)
        return observables

    def get_Hamiltonian(self, opstr_list):
        static = []
        for opstr in opstr_list:
            if len(opstr) == 2:
                static.append([opstr, self.J])
            elif len(opstr) == 1:
                static.append([opstr, self.h])
        dynamic = []
        return hamiltonian(static, dynamic, basis=self.basis)

    def get_var_Hamiltonian(self, opstr_dict):
        op_dict = {}
        for label, opstr_list in opstr_dict.items():
            op_list = []
            for opstr in opstr_list:
                if len(opstr) == 2:
                    op_list.append([opstr, self.J])
                elif len(opstr) == 1:
                    op_list.append([opstr, self.h])
            op_dict[label] = op_list
        return quantum_operator(op_dict, basis=self.basis)

    def get_init_state(self, axis):
        if axis == 'x':
            # all up in x basis
            return (1 / (2 ** (self.N / 2))) * np.ones(2 ** self.N)
        if axis == 'y':
            # all up in y basis
            return (1 / (2 ** (self.N / 2))) * np.array([1. if i % 2 == 0 else 1j for i in range(2 ** self.N)])
    
    def get_squeezing(self, observed_t):
        W_t = np.real((np.array(observed_t['S_plus_S_z']) - np.array(observed_t['S_minus_S_z'])) / 1j)
        V_plus_t = np.array(observed_t['Sy_sq']) + np.array(observed_t['Sz_sq'])
        V_minus_t = np.array(observed_t['Sy_sq']) - np.array(observed_t['Sz_sq'])
        variance_t = (V_plus_t - np.sqrt(np.square(W_t) + np.square(V_minus_t))) / 2
        angle_t = -0.5 * np.arctan(W_t / V_minus_t)
        variance_perp_t = observed_t['Sz_sq'] * (np.cos(angle_t + np.pi / 2)**2) + observed_t['Sy_sq'] * (np.sin(angle_t + np.pi / 2)**2) + np.add(observed_t['Sy_Sz'], observed_t['Sz_Sy']) * np.cos(angle_t + np.pi / 2) * np.sin(angle_t + np.pi / 2)
        min_variance_t = np.array([min(v, v_perp) for v, v_perp in zip(variance_t, variance_perp_t)])
        # normalize to variance of initial state
        norm = variance_t[0]
        min_variance_norm_t = min_variance_t / norm
        opt_angle_t = np.array([angle_t[t] if variance_t[t] <= variance_perp_t[t] else angle_t[t] + np.pi / 2 for t in range(len(angle_t))])
        min_variance_SN_t = np.multiply(min_variance_t, N) / np.square(observed_t['S_x'])
        return min_variance_SN_t, min_variance_norm_t, opt_angle_t

class SpinOperators_Symmetry(object):

    def __init__(self, system_size):
        self.N = system_size[0] * system_size[1]
        
        S = int(self.N / 2)
        Sz_diags = list(reversed(range(- S, S + 1)))
        Sz = diags(Sz_diags)
        Sp_diags = list(reversed([np.sqrt(S * (S + 1) - m * (m + 1)) for m in range(- S, S)]))
        Sp = diags(Sp_diags, 1)
        Sm = diags(Sp_diags, -1)
        Sx = (Sp + Sm) / 2
        Sy = (Sp - Sm) / (2j)
        
        # Operators on pairs of spins (double-counting)
        Sx_sq = Sx * Sx
        Sy_sq = Sy * Sy
        Sz_sq = Sz * Sz

        Sp_Sz = Sp * Sz
        Sm_Sz = Sm * Sz

        Sy_Sz = Sy * Sz
        Sz_Sy = Sz * Sy
        
        self.observables = dict(S_x=Sx, S_y=Sy, S_z=Sz, S_plus=Sp, S_minus=Sm, S_plus_S_z=Sp_Sz, S_minus_S_z=Sm_Sz, Sx_sq=Sx_sq, Sy_sq=Sy_sq, Sz_sq=Sz_sq, Sy_Sz=Sy_Sz, Sz_Sy=Sz_Sy)

    def get_observables(self):
        return self.observables

    def get_Hamiltonian(self, opstr_list, strength_list):
        H = None
        for opstr, strength in zip(opstr_list, strength_list):
            if H is None:
                H = strength * self.observables[opstr]
            else:
                H += strength * self.observables[opstr]
        return H
        
    def get_init_state(self, axis):
        psi_0 = normalize(np.array([1 if i == 0 else 0 for i in range(self.N + 1)]))
        if axis == 'z':
            return psi_0
        if axis == 'x':
            S_y = self.observables['S_y']
            psi_0 = expm_multiply(-1j*S_y*(np.pi / 2), psi_0)
            return psi_0
        if axis == 'y':
            S_x = self.observables['S_x']
            psi_0 = expm_multiply(-1j*(-S_x)*(np.pi / 2), psi_0)
            return psi_0

    def get_squeezing(self, observed_t):
        W_t = np.real((np.array(observed_t['S_plus_S_z']) - np.array(observed_t['S_minus_S_z'])) / 1j)
        V_plus_t = np.array(observed_t['Sy_sq']) + np.array(observed_t['Sz_sq'])
        V_minus_t = np.array(observed_t['Sy_sq']) - np.array(observed_t['Sz_sq'])
        variance_t = (V_plus_t - np.sqrt(np.square(W_t) + np.square(V_minus_t))) / 2
        angle_t = -0.5 * np.arctan(W_t / V_minus_t)
        variance_perp_t = observed_t['Sz_sq'] * (np.cos(angle_t + np.pi / 2)**2) + observed_t['Sy_sq'] * (np.sin(angle_t + np.pi / 2)**2) + np.add(observed_t['Sy_Sz'], observed_t['Sz_Sy']) * np.cos(angle_t + np.pi / 2) * np.sin(angle_t + np.pi / 2)
        min_variance_t = np.array([min(v, v_perp) for v, v_perp in zip(variance_t, variance_perp_t)])
        # normalize to variance of initial state
        norm = variance_t[0]
        min_variance_norm_t = min_variance_t / norm
        opt_angle_t = np.array([angle_t[t] if variance_t[t] <= variance_perp_t[t] else angle_t[t] + np.pi / 2 for t in range(len(angle_t))])
        min_variance_SN_t = np.multiply(min_variance_t, self.N) / np.square(observed_t['S_x'])
        return min_variance_SN_t, min_variance_norm_t, opt_angle_t

class SpinOperators_DTWA(object):

    def __init__(self, structure, system_size, fill, coord=[]):
        self.N = system_size[0] * system_size[1]
        if list(coord) == []:
            if system_size[0] == 1 or system_size[1] == 1:
                print('1D')
                self.coord = dtwa_util.getLatticeCoord(1,self.N,1)
            elif system_size[0] == system_size[1]:
                print('2D')
                self.coord = dtwa_util.getLatticeCoord(2,system_size[0],1)
            else:
                raise Exception('Invalid system size! System size must be 1 x L, L x 1, or L X L.')
        else:
            self.coord = coord
        self.nt =  10**4  # number of init conditions = number of trajectories

    def get_Ising_Hamiltonian(self, J, alpha, Jij=[]):
        def evolve(configs, tvec):
            # Ising interaction for interacting spins (no double-counting)
            tdist, meanConfig_evol = dtwa.IsingEvolve(configs,tvec,J,coord=self.coord,Jfunc=[],alpha=alpha,Jij=Jij)
            return tdist
        return evolve
    
    def get_transverse_Hamiltonian(self, axis):
        def evolve(configs, tvec):
            tdist, meanConfig_evol = dtwa.TFEvolve(configs,tvec,1., axis=axis)
            return tdist
        return evolve

    def get_XXZ_Hamiltonian(self, Jz, Jperp, alpha, Jij=[]):
        def evolve(configs, tvec):
            # XXZ interaction for interacting spins (no double-counting)
            tvec = np.concatenate(([0], tvec))
            tdist, meanConfig_evol = dtwa.XXZEvolve(configs,tvec,Jz,Jperp,coord=self.coord,Jfunc=[],alpha=alpha,Jij=Jij)
            tdist = np.swapaxes(tdist,0,1)
            return tdist[1:]
        return evolve

    def get_XY_Hamiltonian(self, Jperp, alpha, Jij=[]):
        def evolve(configs, tvec):
            tvec = np.concatenate(([0], tvec))
            tdist, meanConfig_evol = dtwa.XYEvolve(configs,tvec,Jperp,coord=self.coord,Jfunc=[],alpha=alpha,Jij=Jij)
            tdist = np.swapaxes(tdist,0,1)
            return tdist[1:]
        return evolve
    
    def get_TFI_Hamiltonian(self, Jz, h, alpha, Jij=[]):
        def evolve(configs, tvec):
            tvec = np.concatenate(([0], tvec))
            tdist, meanConfig_evol = dtwa.TFIEvolve(configs,tvec,Jz,h,coord=self.coord,Jfunc=[],alpha=alpha,Jij=Jij)
            tdist = np.swapaxes(tdist,0,1)
            return tdist[1:]
        return evolve

    def get_CT_Hamiltonian(self, J, alpha, Jij=[]):
        def evolve(configs, tvec):
            tvec = np.concatenate(([0], tvec))
            tdist, meanConfig_evol = dtwa.CTEvolve(configs,tvec,J,coord=self.coord,Jfunc=[],alpha=alpha,Jij=Jij)
            tdist = np.swapaxes(tdist,0,1)
            return tdist[1:]
        return evolve

    def get_CTv2_Hamiltonian(self, J, alpha, Jij=[]):
        def evolve(configs, tvec):
            tvec = np.concatenate(([0], tvec))
            tdist, meanConfig_evol = dtwa.CTv2Evolve(configs,tvec,J,coord=self.coord,Jfunc=[],alpha=alpha,Jij=Jij)
            tdist = np.swapaxes(tdist,0,1)
            return tdist[1:]
        return evolve

    def get_init_state(self, axis):
        return dtwa.genUniformConfigs(self.N,self.nt,axis=axis)

    def get_observed(self, tdist, meanConfig_evol):
        # get mean total spin components and their squares
        Stot_dtwa = np.sum(meanConfig_evol,axis=1).real.transpose()
        Ssq_dtwa = np.mean(np.sum(tdist,axis=2)**2,axis=1).real.transpose()

        Sx_dtwa, Sy_dtwa, Sz_dtwa = Stot_dtwa
        Sx_sq_dtwa, Sy_sq_dtwa, Sz_sq_dtwa = Ssq_dtwa
        return dict(S_x=Sx_dtwa, S_y=Sy_dtwa, S_z=Sz_dtwa, Sx_sq=Sx_sq_dtwa, Sy_sq=Sy_sq_dtwa, Sz_sq=Sz_sq_dtwa)

    def get_squeezing(self, tdist, meanConfig_evol):
        # get mean total spin components and their squares
        Stot_dtwa = np.sum(meanConfig_evol,axis=1).real.transpose()
        Ssq_dtwa = np.mean(np.sum(tdist,axis=2)**2,axis=1).real.transpose()

        # obtain quantities used to calculate the squeezing parameter
        # W = < {Sy,Sz} > = expectation value of anticommutator
        # V± = < Sy^2 ± Sz^2 > = total variance in orthogonal plane (since mean total spin component always points along x) 
        W_dtwa = np.array([dtwa.getW(dist) for dist in tdist]) 
        Vp_dtwa = np.array([dtwa.getVpm(dist,'p') for dist in tdist])
        Vm_dtwa = np.array([dtwa.getVpm(dist,'m') for dist in tdist])

        squeezing_dtwa = (Vp_dtwa - np.sqrt(W_dtwa**2 + Vm_dtwa**2))/2
        opt_angle_t = -0.5 * np.arctan(W_dtwa / Vm_dtwa)

        # normalize to variance of initial state
        norm = squeezing_dtwa[0]
        min_variance_norm_t = squeezing_dtwa / norm

        Sx_sq_dtwa = np.array([np.mean(np.sum(dist[:,:,0],axis=1))**2 for dist in tdist])
        min_variance_SN_t = np.multiply(squeezing_dtwa, self.N) / Sx_sq_dtwa

        return min_variance_SN_t, min_variance_norm_t, opt_angle_t