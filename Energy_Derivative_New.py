#!/usr/bin/env python

import numpy as np
from scipy.spatial import distance as sd
import tomli

with open("input.toml", "rb") as f:
    input = tomli.load(f)

C_dd = input["simulation_properties"]["dipole_moment"]   # Cdd/4pi
w_z = input["simulation_properties"]["trapping_frequency_z"]
w_p = input["simulation_properties"]["trapping_frequency_transverse"]
e = input["simulation_properties"]["dipole_unit_vector"]
m = input["simulation_properties"]["mass"]
k = input["simulation_properties"]["wall_repulsion_coefficient"]
rep_order = -input["simulation_properties"]["order_repulsive_wall"] # e.g -6 or -12
H = k
e_i = np.array(e,dtype = float)  
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
    total_dist_3=r_ij**(-3/2)  # calculates |r_ij|^-3 for each combination of i and j
    np.fill_diagonal(total_dist_3, 0) # changes 1's on diagonals back to 0's (diagonal is r_11 etc. always 0)  

    V_dd_1 = np.sum(total_dist_3)/2 # Calulates the sum of all rij^-3 and divides by 2 to avoid double counting 


    total_dist_5=r_ij**(-5/2) # calculates |r_ij|^-5 for each combination of i and j
    np.fill_diagonal(total_dist_5, 0) # changes 1's on diagonals back to 0's (diagonal is r_11 etc. always 0)  
    dis_e = np.sum(dist_vect*e_hat, axis = 2)**2 # square of e_hat dot r_ij
    # print(dist_vect*e_hat)
    V_dd_2 = (np.sum(dis_e*total_dist_5)/2) # sum of all the previous terms divide by 2 to stop double counting

    V_dd = (C_dd)*(V_dd_1-3*V_dd_2) 
    
    return V_dd

def V_repulsive(R):
    """ Calculates the potential energy of the system due to a r_ij^12 repulsive potential between each particle
    Parameters
    ----------
        R: 2D numpy array - shape (N,3)
    """
    
    sp_result = sd.pdist(R)**(rep_order) # Array of distances between particle^-12 (for 3 particles [r12^-12,r13^-12,r23^-12])
    V_rep = np.sum(sp_result)
    return H*V_rep


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
    """ Returns an array of the derivatives of the trapping potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_rep])
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    """
    V_trap_dx = (trap_f_array_dx*R).flatten()
    return V_trap_dx

def V_rep_dx(dist_vect, dist_2):
    """ Returns an array of the derivatives of the repulsive potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_rep])
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    # """

    r_ij = np.sum(dist_2,axis = 2) # converts to 2D np array  - sums the differences between particles in x y and z diresction [(x_1-x_2)^2, (y_1-y_2)^2, (z_1-z_2)^2]  ->  (x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2 

    np.fill_diagonal(r_ij, 1)
    total_dist_5=r_ij**((rep_order-2)/2) # calculates |r_ij|^-rep for each combination of i and j
    np.fill_diagonal(total_dist_5, 0)

    X = dist_vect.transpose().transpose((0,2,1))
    final = X*total_dist_5

    V_rep_dx = np.sum(final,axis = 2).transpose().flatten()
    return (rep_order)*H*V_rep_dx


def V_dd_dx(dist_vect, dist_2):

    """ Returns an array of the derivatives of the dipolar interaction potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_rep])
    Parameters  
    ----------
        dist_vect: 2D numpy array - shape (N,3)
    """
    r_ij = np.sum(dist_2,axis = 2)

    np.fill_diagonal(r_ij, 1)
    total_dist_5=r_ij**(-5/2)
    np.fill_diagonal(total_dist_5, 0)

    X = dist_vect.transpose().transpose((0,2,1))
    final = X*total_dist_5
    p1 = np.sum(final,axis = 2).transpose().flatten()


    e_dot_dist = np.sum(dist_vect*e_hat,axis=2)
    e_dot_dist_2 = e_dot_dist**2
    total_dist_7 = r_ij**(-7/2)
    np.fill_diagonal(total_dist_7,0)
    ed2_td7 = e_dot_dist_2*total_dist_7
    X = dist_vect.transpose().transpose((0,2,1))
    final_2 = X*ed2_td7
    p2 = -5*np.sum(final_2,axis = 2).transpose().flatten()

    ed_td5 = e_dot_dist*total_dist_5
    E_mat = np.repeat(e_hat,ed_td5.shape[0]**2).reshape(3,ed_td5.shape[0],ed_td5.shape[0])
    final_3 = E_mat*ed_td5
    p3 = 2*np.sum(final_3,axis = 2).transpose().flatten()

    V_dd_dx = -3*C_dd*(p1+p2+p3)

    return V_dd_dx


def V_total_dx_array(x0):
    #R = np.reshape(x0, (x0.shape[0]//3, 3)) ### !!!!!! Check!
    R = np.array(np.split(x0,len(x0)/3), dtype=float) # need to fix
    dist_vect = R.reshape(R.shape[0], 1, 3) - R # Subtracts all particle postions from all others gives displacement vectors in a np array (np.newaxis increases the dimension of the array from 2 -> 3) 
    dist_2 = dist_vect**2 # square of distances 3D
    V_dx_array = V_trap_dx(R)   + V_rep_dx(dist_vect, dist_2)  + V_dd_dx(dist_vect, dist_2)
    return V_dx_array
