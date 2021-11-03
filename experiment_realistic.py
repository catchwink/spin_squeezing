# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:09:28 2020

@author: Gabriel Moreau
email: moreauga@stanford.edu
version: 1.2
Implementation of the analytical solutions for calculating spin squeezing
in Foss-Feig et al, PHYSICAL REVIEW A 87, 042101 (2013).

"""
import numpy as np
from random import gauss
from scipy import constants
from time import perf_counter
import multiprocessing as mp
import pandas as pd
from bisect import bisect_left
from scipy import interpolate
#import matplotlib.pyplot as plt
#from numba import njit, prange
#from multiprocessing.dummy import Pool as ThreadPool


class SqueezingCalculations:
    def __init__(self,N_atoms=1000,interaction_type="ARC_full",detuning=20e6,rabi_freq=2e6,C_6=100e9,x_rad=6e-6,y_rad=6e-6,z_rad=15e-6,nCores=8,spin_chain=False,distribution="gaussian",spacing = 2e-6,arcFileName=None,val_at_inf=None):
        # ADD DESCRIPTION
        # Read in the fit results of the ARC potential
        if interaction_type == "ARC_interpolate_theta":
            fileName = "arcDataFit.csv" 
            fileNameIn = "G:\\Shared drives\\Rydberg Drive\\Gabriel\\Squeezing Theory\\DataSets\\arcResults\\" + fileName
            dataIn = pd.read_csv(fileNameIn)
            self.data = dataIn.to_numpy()
            self.theta_list = self.data.T[1]
            self.data = np.delete(self.data,(0,1),axis=1)
        elif (interaction_type == "ARC_full"):
            # See wiki entry for 2021/5/29-30 
            if arcFileName:
                fileName = arcFileName
                if not val_at_inf:
                    raise Exception("ARC input file provided without val_at_inf")
            elif detuning == 20e6 :
                fileName = "arcData30-05-2021_14-58-52_Det_20.0_MHz__rabi_2.70_MHz_n_43.csv"
                val_at_inf = 7635.488835
            elif detuning == 8e6:
                fileName = "arcData30-05-2021_23-09-58_Det_8.0_MHz__rabi_1.5_MHz_n_60.csv"
                val_at_inf = 23407.392
            else:
                raise Exception('Interaction type "ARC_full" requested but no input file associated with a detuning of' + str(detuning*1e-6) + 'MHz')
            # val_at_inf = 2298.803091 # Calculated at r=1micron, in dataset arcData20-05-2021_15-20-26_Det_20.0_MHz_n_43.csv
            # fileName = "arcData_Detuning_20MHz_Combined.csv" # Fine sweep over r and theta at 20MHz, with an extra fine sweep in the region where the asymptote is present
            fileNameIn = "G:\\Shared drives\\Rydberg Drive\\Gabriel\\Squeezing Theory\\DataSets\\arcResults\\" + fileName
            dataIn = pd.read_csv(fileNameIn)
            self.data = dataIn.to_numpy()
            self.r_list = self.data[0][2:]
            self.theta_list = self.data.T[1][1:]
            self.data = np.delete(self.data,(0,1),axis=1)
            self.data = np.delete(self.data,0,axis=0)
            self.data = self.data - val_at_inf # Substract offset so potential is 0 at inf
            self.interpolatedPotential = interpolate.interp2d(self.r_list,self.theta_list, self.data)
        self.nCores = nCores
        self.hbar = constants.hbar
        self.N_atoms = N_atoms
        self.interaction_type = interaction_type
        self.C_6 = C_6*1e-36 # Convert C6 coefficient from Hz*(micron)^6 to Hz*(m)^6
        self.rc = (self.C_6/(2*detuning))**(1/6)
        self.J_0 = -2*np.pi*(rabi_freq)**4/(8*(detuning)**3) 
        self.x_rad, self.y_rad, self.z_rad = x_rad, y_rad, z_rad 
        self.spin_chain = spin_chain
        self.distribution = distribution
        self.spacing = spacing
        self.setup(spacing = self.spacing)
        self.makeInteractionMatrix()
        self.correlationVals = np.zeros((self.N_atoms,self.N_atoms),dtype=object)
        return None
        
    def setup(self,density=1e17,spacing=2e-6):
        """
        Setup list with coordinates of atoms in a sphere at random coordinates
            * For now, this assumes atoms can be arbitrarily close to one another
            
        """
        mu = 0 # Mean of random distribution of atom positions
        self.points = np.zeros((self.N_atoms,3))
        averageSpacing = (4*np.pi*density/3)**(-1/3)
        if self.spin_chain:
            if self.distribution == "square lattice":
                for i in range(self.N_atoms):
                    self.points[i,:] = (i-self.N_atoms/2)*averageSpacing, 0, 0
            elif self.distribution == "gaussian":
                for i in range(self.N_atoms):
                    self.points[i,:] = gauss(0,self.x_rad), 0, 0
            return(self.points)
        else:
            if self.distribution == "square lattice":
                N = 2*int(self.N_atoms**(1/3))
                index=0
                squareLattice = np.zeros((8*N**3,3))
                for i in np.arange(-N,N):
                    for j in np.arange(-N,N):
                        for k in np.arange(-N,N):
                            squareLattice[index,:] = i,j,k
                            index += 1
                squareLattice = np.array(squareLattice)
                n = 0
                r = 0
                while (n<self.N_atoms):
                    i=0
                    while(i<len(squareLattice)):
                        if ((squareLattice[i,0]**2 + squareLattice[i,1]**2 + squareLattice[i,2]**2) <= r**2):
                            if n == self.N_atoms:
                                break
                            self.points[n] = squareLattice[i]
                            n +=1
                            squareLattice = np.delete(squareLattice,i,axis=0) # Don't keep adding these points
                            # Uncoment following 2 lines for progress updates
                            #if n / self.N_atoms*1000 %10 == 0:
                            #    print(f"{int(n/self.N_atoms*100)} % progress")
                        i +=1
                    r += 0.1
                self.points *= spacing
            elif self.distribution == "gaussian":
                for i in range(self.N_atoms):
                    self.points[i,:] = gauss(mu,self.x_rad),gauss(mu,self.y_rad),gauss(mu,self.z_rad)        
            return(self.points)


    def interaction(self,r1,r2):
        # Calculate the distance between two atoms
        dist = np.sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2 )
        if self.interaction_type == "rawDist":
            return dist
        # Calculate and return interaction strength for all to all interactions
        if self.interaction_type == "ATA":
            # Need to figure out an implement this interaction type
            return self.J_0
        elif self.interaction_type == "RD":
            # Rydberg dressed interaction
            J = self.J_0 / (1 + (dist/self.rc)**6)
            return J
        elif self.interaction_type == "ARC_fit":
            # Calculate theta from dot product:
            quant_axis = [1,0,0]
            r = [r1[0]-r2[0],r1[1]-r2[1],r1[2]-r2[2]]
            theta = np.arccos(np.dot(r,quant_axis)/(np.linalg.norm(r)*np.linalg.norm(quant_axis)))
            # The numbers bellow come from the results of the ARC potential fits
            a = 1457.4 * np.cos(2.0029*theta) - 1473.8
            b = -0.09871 * np.cos(1.9912 * theta) + 4.8653
            c = -1461.3 * np.cos(2.0029 * theta) - 3503.0
            d = -0.01621 * np.cos(1.9942 * theta) + 1.5358
            J =  ( a /(1+((dist*1e6)/b)**6) + c/(1+((dist*1e6)/d)**6)) # 2*np.pi * Fits were done in units of microns
            return J
        elif self.interaction_type == "ARC_interpolate_theta":
            quant_axis = [1,0,0]
            r = [r1[0]-r2[0],r1[1]-r2[1],r1[2]-r2[2]]
            theta = np.arccos(np.dot(r,quant_axis)/(np.linalg.norm(r)*np.linalg.norm(quant_axis)))
            # Find 2 closest values
            closest_theta_index = bisect_left(self.theta_list,theta)
            if theta == self.theta_list[closest_theta_index]:
                # Deal with the very unlikely case that a randomly generated theta is exactly the same as one in the theta_list
                params = self.data[closest_theta_index]
            elif closest_theta_index == 0:
                params = self.data[0]
            else:
                # Perform linear interpolation
                params = np.zeros(4)
                x_vals = [self.theta_list[closest_theta_index-1],self.theta_list[closest_theta_index]]
                #print(f"\ntheta-: {x_vals[0]:.5f}, theta: {theta:.5f}, theta+: {x_vals[1]:.5f}")
                for i in range(len(params)):
                    params[i] = np.interp(theta,x_vals,[self.data[closest_theta_index-1][i],self.data[closest_theta_index][i]])
                    #print(f"Param {i}: {self.data[closest_theta_index-1][i]:.5f}, {params[i]:.5f}, {self.data[closest_theta_index][i]:.5f}")
            J =  (params[0] /(1+((dist*1e6)/params[1])**6) + params[2]/(1+((dist*1e6)/params[3])**6))
            
            return J
        elif self.interaction_type == "ARC_full":
            quant_axis = [1,0,0]
            r = [r1[0]-r2[0],r1[1]-r2[1],r1[2]-r2[2]]
            theta = np.arccos((r[0]*quant_axis[0]+r[1]*quant_axis[1]+r[2]*quant_axis[2])/(np.sqrt(r[0]**2+r[1]**2+r[2]**2)*np.sqrt(quant_axis[0]**2+quant_axis[1]**2+quant_axis[2]**2)))
            if theta > np.pi/2:
                theta = np.pi - theta
            # Find 2 closest values
            J = self.interpolatedPotential(dist*1e6,theta)[0]
            return J
        else:
            print("Invalid interaction type!")
            quit()
    
    def makeInteractionMatrix(self):
        self.interactionMatrix = np.zeros((self.N_atoms,self.N_atoms))
        for i in range(self.N_atoms):
            for j in range(i,self.N_atoms):
                # Interaction is 0 if i=j
                if i == j:
                    self.interactionMatrix[i,j] = 0
                else:
                    interaction_value = self.interaction(self.points[i,:], self.points[j,:])
                    self.interactionMatrix[i,j] = interaction_value
                    self.interactionMatrix[j,i] = interaction_value
        return self.interactionMatrix
        
    
    def calculateSqueezing(self,t):
        
        # np.newaxis is here to put the vector into (1, N) shape
        cosProduct = np.prod(np.cos(t*self.interactionMatrix), axis=0)[np.newaxis, :]
    
        zpCorr = 0.25*(np.sin(self.interactionMatrix*t) * cosProduct /
                                 np.cos(t*self.interactionMatrix))
        zmCorr = -zpCorr
        # only works for symmetric matrices
    
        # each element m, n of col_matrix will be column n of interactionMatrix
        #col_matrix = np.tile(self.interactionMatrix.T[np.newaxis, :], (self.N_atoms, 1, 1))
        col_matrix = self.interactionMatrix.T[np.newaxis, :]
        # each element m, n of row_matrix will be row m of interactionMatrix
        #row_matrix = np.tile(self.interactionMatrix[:, np.newaxis, :], (1, self.N_atoms, 1))
        row_matrix = self.interactionMatrix[:, np.newaxis, :]
        pmCorr = 0.25*np.prod(np.cos(t*(col_matrix - row_matrix)), axis=2)
        np.fill_diagonal(pmCorr, 0.5)
        ppCorr = 0.25*np.prod(np.cos(t*(col_matrix + row_matrix)), axis=2)
        np.fill_diagonal(ppCorr,0) # Diag should actually be 0 as s+s+ = 0 for m = n
        xxCorr = 0.5*(pmCorr + ppCorr)
        yyCorr = 0.5*(pmCorr - ppCorr)
        zzCorr = 0.25*np.identity(self.N_atoms)
        
        return np.array([zpCorr,zmCorr,pmCorr,ppCorr,xxCorr,yyCorr,zzCorr]) 
    
    
    def calcSx(self,t):
        sx = 0.5*np.ones(self.N_atoms)
        for n in range(self.N_atoms):
            sx[n] *= np.prod(np.cos(t*self.interactionMatrix[n,:]))
        return sx
    
    def multiProcessing(self,t_list):
        # Implement multiprocessing
        cores = self.nCores # Choose # of cores here! (otherwise only 1 core used as a precaution to avoid filling up entire memory)
        if cores == 0:
            cores = 1
        t0 = perf_counter()
        t_length = len(t_list)
        Xvar = np.zeros(t_length)
        Yvar = np.zeros(t_length)
        Zvar = np.zeros(t_length)
        w = np.zeros(t_length)
        sx = np.zeros(t_length)
        sx_av = np.zeros(t_length)
        
        with mp.Pool(processes=cores) as p:
            results = p.map(self.calculateSqueezing, t_list)
            sx = p.map(self.calcSx, t_list)
        for i in np.arange(t_length):
            w[i] = np.sum(results[i][0] - results[i][1]) # <sZs+> - <sZs->, As original corr functions are purely imagenary, don't need to worry about dividing by j and taking real part
            #Xvar[i] = np.sum(sx[i])
            Xvar[i] = np.sum(results[i][4])-(np.sum(sx[i]))**2 # <sXsX> - <sX>^2
            Yvar[i] = np.sum(results[i][5])
            Zvar[i] = np.sum(results[i][6])
            sx_av[i] = np.average(sx[i])
            #sx[i] = results[i][7]
        vP = Yvar + Zvar
        vM = Yvar - Zvar
        alphaVar = 0.5*(vP-np.sqrt(w**2+vM**2))
        alphaMin = -0.5*np.arctan(w/vM)
        sqzngPar = self.N_atoms * alphaVar/ sx_av**2/( self.N_atoms**2)
        sx_sum = np.sum(sx,axis=1) # Total sx as a function of time
        tf = perf_counter()
        #Old definition of Q: 
        Q = np.average(np.sum(self.interactionMatrix,axis=0))
        
        # for i in range(len(self.interactionMatrix)):
            
        print(f"Multithreaded computing time: {tf-t0:.3f} s")
        return Xvar,Yvar,Zvar,alphaMin,alphaVar,sqzngPar,sx_sum,Q
if __name__ == "__main__":
    squeezing = SqueezingCalculations(interaction_type="ARC_full")