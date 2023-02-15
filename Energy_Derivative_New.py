#!/usr/bin/env python

import numpy as np
from scipy.spatial import distance as sd
import tomli

with open("input.toml", "rb") as f:
    variables = tomli.load(f)

C_dd = variables["simulation_properties"]["dipole_moment"]   # sqrt(Cdd/4pi)
w_z = variables["simulation_properties"]["trapping_frequency_z"]
w_p = variables["simulation_properties"]["trapping_frequency_transverse"]
e = variables["simulation_properties"]["dipole_direction_vector"]
m = variables["simulation_properties"]["mass"]
c_6 = variables["simulation_properties"]["wall_repulsion_coefficient"]
rep_order = -variables["simulation_properties"]["order_repulsive_wall"] # e.g -6 or -12
e_i = np.array(e,dtype = float)

# Trapping errors of input variables:
if e_i.any() == np.array([0.0,0.0,0.0]).any():
    raise ValueError("magnitude of the dipole_direction_vector cannot equal 0" )
if C_dd < 0:
    raise ValueError("dipole moment must be positive")
if w_z < 0:
    raise ValueError("trapping_frequency_z must be positive")
if w_p < 0:
    raise ValueError("trapping_frequency_transverse must be positive")
if rep_order> 0:
    raise ValueError("order_repulsive_wall mus be positive (e.g 6 or 12)")
if m< 0 :
    raise ValueError("mass must be positive")

e_hat = e_i / np.linalg.norm(e_i) # Unit vector of dipole orientation

trap_f_array = np.array([(w_p**2)*(m/2),(w_p**2)*(m/2),(w_z**2)*(m/2)])
trap_f_array_dx = np.array([(w_p**2)*(m),(w_p**2)*(m),(w_z**2)*(m)])

# Calculates the potential of dipoles in a quadratic trapping potential:

def V_trap(R):   
    """ Calculates the trapping potential of the system
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    """
    V_trap = np.dot(trap_f_array,np.sum(np.square(R),axis = 0))    # dot product of trapping frequencies and sum of all x^2,y y^2, z^2
    return V_trap

def V_dd(R):
    """ Calculates the potential energy of the system due to the dipole interactions between each particle
    Parameters
    ----------
        R: 2D numpy array - shape (N,3)
    """
    dist_vect = R.reshape(R.shape[0], 1, 3) - R # Subtracts all particle postions from all others gives displacement vectors in a np array so that you get 3D array of shape (N,N,3) [[[x_1-x_1, y_1-y_1, z_1-z_1],[x_1-x_2, y_1-y_2, z_1-z_2]...]]
    dist_2 = dist_vect**2 # Takes the square of the difference between each coordinates e.g (x_1 - x_2) -> (x_1 - x_2)^2
    r_ij = np.sum(dist_2,axis = 2) # converts to 2D np array  - sums the differences between particles in x y and z diresction [(x_1-x_2)^2, (y_1-y_2)^2, (z_1-z_2)^2]  ->  (x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2 

    np.fill_diagonal(r_ij, 1) # adds 1's to diagonal to avoid divide by 0 errors
    total_dist_3=r_ij**(-1.5)  # calculates |r_ij|^-3 for each combination of i and j
    np.fill_diagonal(total_dist_3, 0) # changes 1's on diagonals back to 0's (diagonal is r_11 etc. always 0)  

    V_dd_1 = np.sum(total_dist_3)*0.5 # Calulates the sum of all rij^-3 and divides by 2 to avoid double counting 


    total_dist_5=r_ij**(-5/2) # calculates |r_ij|^-5 for each combination of i and j
    np.fill_diagonal(total_dist_5, 0) # changes 1's on diagonals back to 0's (diagonal is r_11 etc. always 0)  
    dis_e = np.sum(dist_vect*e_hat, axis = 2)**2 # square of e_hat dot r_ij
    # print(dist_vect*e_hat)
    V_dd_2 = (np.sum(dis_e*total_dist_5)*0.5) # sum of all the previous terms divide by 2 to stop double counting

    V_dd = (C_dd**2)*(V_dd_1-3*V_dd_2) 
    
    return V_dd

def V_repulsive(R):
    """ Calculates the potential energy of the system due to a r_ij^12 repulsive potential between each particle
    Parameters
    ----------
        R: 2D numpy array - shape (N,3)
    """
    
    sp_result = sd.pdist(R)**(rep_order) # Array of distances between particle^-12 (for 3 particles [r12^-12,r13^-12,r23^-12])
    V_rep = np.sum(sp_result)
    return c_6*V_rep


def V_total(x0):
    """ Calculates the total potential energy of a system of dipolar particles
    Parameters  
    ----------
        x0: 1D numpy array - shape (N*3)  
                           - [x1, y1, z1, ..., xN, yN, zN]
    """

    R = np.array(np.split(x0,len(x0)/3), dtype=float)  # Splitting x0 into a 2D np array - [[x1, y1, z1],...,[xN, yN, zN]]

    V_tot = V_repulsive(R) + V_dd(R)  + V_trap(R)  # Sum of all potential terms
    return V_tot


# Derivative Arrays:

def V_trap_dx(R):
    """ Returns an array of the derivatives of the trapping potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_trap])
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    """
    V_trap_dx = (trap_f_array_dx*R).flatten()
    return V_trap_dx

def V_rep_dx(dist_vect, r_ij):
    """ Returns an array of the derivatives of the repulsive potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_rep])
    Calculates:

    each term is dV_rep/dx_k = Sum_i(+/-(rep_order)(x_k-x_i)r_ik^(rep_order-2)) where if k>i its -, if k<i its + 
    
    
    Parameters  
    ----------
        dist_vect: 3D numpy array - shape (N,N,3)
        dist_2: 3D numpy array - shape (N,N,3)
    """


    np.fill_diagonal(r_ij, 1) # adds 1's to diagonal to avoid divide by 0 errors
    total_dist_5=r_ij**((rep_order-2)*0.5) # calculates |r_ij|^-rep for each combination of i and j
    np.fill_diagonal(total_dist_5, 0) # changes 1's on diagonals back to 0's (diagonal is r_11 etc. always 0)

    X = dist_vect.transpose().transpose((0,2,1))  # Converting to a 3d np array (3, N, N) with [[x diplacements (NxN)], [y displacements (NxN)], [z displacements (NxN)]]
    final = X*total_dist_5  # Each of 2x2 displacement arrays * arrays of r_ij^-5 

    V_rep_dx = np.sum(final,axis = 2).transpose().flatten()  
    return (rep_order)*c_6*V_rep_dx


def V_dd_dx(dist_vect, r_ij):

    """ Returns an array of the derivatives of the dipolar interaction potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_dd])
  
    Each element in the array calculates
    dV_dd/dx_k = Sum_i((+/-(x_k-x_i)*(|r_ij|^2-5*(e dot r_kj)^2) + 2*e_k*(e dot rkj)*|r_kj|^2)/(|r_kj|^7)) where if k>i its -, if k<i its + (e_k is the directional component of e_hat)
    
    Parameters    
    ----------
        dist_vect: 3D numpy array - shape (N,N,3)
        dist_2: 3D numpy array - shape (N,N,3)
    """

    np.fill_diagonal(r_ij, 1) # adds 1's to diagonal to avoid divide by 0 errors
    total_dist_5=r_ij**(-2.5)
    np.fill_diagonal(total_dist_5, 0) # changes 1's on diagonals back to 0's (diagonal is r_11 etc. always 0)

    X = dist_vect.transpose().transpose((0,2,1)) # Converting to a 3d np array (3, N, N) with [[x diplacements (NxN)], [y displacements (NxN)], [z displacements (NxN)]]
    final = X*total_dist_5
    p1 = np.sum(final,axis = 2).transpose().flatten()


    e_dot_dist = np.sum(dist_vect*e_hat,axis=2)
    e_dot_dist_2 = e_dot_dist**2
    total_dist_7 = r_ij**(-3.5)

    np.fill_diagonal(total_dist_7,0) # changes 1's on diagonals back to 0's (diagonal is r_11 etc. always 0)
    ed2_td7 = e_dot_dist_2*total_dist_7
    final_2 = X*ed2_td7
    p2 = -5*np.sum(final_2,axis = 2).transpose().flatten()

    ed_td5 = e_dot_dist*total_dist_5
    E_mat = np.repeat(e_hat,ed_td5.shape[0]**2).reshape(3,ed_td5.shape[0],ed_td5.shape[0])
    final_3 = E_mat*ed_td5
    p3 = 2*np.sum(final_3,axis = 2).transpose().flatten()

    V_dd_dx = -3*C_dd**2*(p1+p2+p3)

    return V_dd_dx


def V_total_dx_array(x0):

    """ Returns an array of the derivatives of the repulsive potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_total])
 
    Parameters    
    ----------
        x0: 1D numpy array of atomic positions [x1,y1,z1,...,x_N,y_N,z_N] - shape (3N)
    """

    #R = np.reshape(x0, (x0.shape[0]//3, 3)) ### !!!!!! Check!
    R = np.array(np.split(x0,len(x0)/3), dtype=float) # need to fix
    dist_vect = R.reshape(R.shape[0], 1, 3) - R # Subtracts all particle postions from all others gives displacement vectors in a np array (np.newaxis increases the dimension of the array from 2 -> 3) 
    dist_2 = dist_vect**2 # square of distances 3D
    r_ij = np.sum(dist_2,axis = 2)
    V_dx_array = V_trap_dx(R) + V_rep_dx(dist_vect, r_ij) + V_dd_dx(dist_vect, r_ij)
    return V_dx_array
