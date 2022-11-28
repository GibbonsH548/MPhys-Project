import numpy as np
import pandas as pd
import scipy.spatial.distance as sd

import warnings
#suppress warnings (to stop printing divide by 0 error)
warnings.filterwarnings('ignore')

# hbar, M - atom, 1/w

# Constants:
# Real values 
m_i = 2.72328406922e-25  # Mass of particles (kg)

mu_0_i = 1.257*10**(-7)  # Vacuum permability ()
mu_b_i = 9.27400968e-24  # Bohr magneton ()
mu_m = 10*mu_b_i         # magnetic moment of particles

e_i = np.array([0,0,1])  # Unit vector of dipole orientation

# trapping frequency: If w_p << w_z - pancake if w_p >> w_z - cigar
w_p_i = 1               # axial trapping frequency
w_z_i = 50          # radial trapping frequency
# Too big problem with derivatives?

# Converting to sensible units 
kg = m_i**(-1)
m = m_i*kg     # convert to mass = 1

w_2_i = w_p_i**2 + w_z_i**2   # converting to w**2 = 1

Hz = (w_2_i**(-1/2))        # Converting Hz 
w_p = w_p_i*Hz              # converting axial trapping frequency
w_z = w_z_i*Hz              # converrting radial trapping frequency

mu_0 = mu_0_i*kg*Hz**(2)    # converting vacuum permabilty to new units


C_dd = mu_0*(mu_m)**2    # C_dd is the the coupling constant  - μ_0 μ_m^2 


trap_f_array = np.array([(w_p**2)*(m/2),(w_p**2)*(m/2),(w_z**2)*(m/2)])
trap_f_array_dx = [(w_p**2)*(m),(w_p**2)*(m),(w_z**2)*(m)]


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
    dist_vect = (R[np.newaxis, :, :] - R[:, np.newaxis, :]) # Subtracts all particle postions from all others gives displacement vectors in a np array (np.newaxis increases the dimension of the array from 2 -> 3) 

    distance_array = sd.pdist(R)   # Array of distances between particles (for 3 particles [r12,r13,r23])
    V_dd_1 = np.sum(distance_array**-3) # Calulates the sum of all rij^-3
    e_dot_rij_sqr_wz = np.tril(np.dot(dist_vect,e_i)**2).flatten() # Produces an array of the dot products or the dipolar unit vector with the diplacement between particles 
    e_dot_rij_sqr = e_dot_rij_sqr_wz[e_dot_rij_sqr_wz!=0] # Removes the zeros from duplicates and rij where i = j (ie)

    V_dd_2 = np.dot(distance_array**-5,e_dot_rij_sqr) # Calculates the dot product of rij^-5 with (e dot rij)^2 which finds the total potential from all particles 
    V_dd = (C_dd/(4*np.pi))*(V_dd_1-3*V_dd_2) 
    return V_dd

def V_repulsive(R):
    """ Calculates the potential energy of the system due to a r_ij^12 repulsive potential between each particle
    Parameters
    ----------
        R: 2D numpy array - shape (N,3)
    """
    sp_result = sd.pdist(R)**(-12) # Array of distances between particle^-12 (for 3 particles [r12^-12,r13^-12,r23^-12])
    V_rep = np.sum(sp_result)
    return V_rep

def V_total(x0):
    """ Calculates the total potential energy of a system of dipolar particles
    Parameters  
    ----------
        x0: 1D numpy array - shape (N*3)  
                           - [x1, y1, z1, ..., xN, yN, zN]
    """
    R = np.array(np.split(x0,len(x0)/3))  # Splitting x0 into a 2D np array - [[x1, y1, z1],...,[xN, yN, zN]]
    V_tot = V_trap(R)+V_repulsive(R)+V_dd(R)   # Suming over all contributions to the overall potential energy
    return V_tot


# Derivative Array:

def V_rep_dx(R):
    """ Returns an array of the derivatives of the repulsive potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_rep])
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    """

    # array of distances between particles 
    # [[[x00,y00,z00],[x01,y01,z01]],[[x10,y10,z10],[x11,y11,z11]]]
    dist_vect = (R[np.newaxis, :, :] - R[:, np.newaxis, :]) # Subtracts all particle postions from all others gives displacement vectors in a np array (np.newaxis increases the dimension of the array from 2 -> 3) 

    # square of distances 3D
    dist_2 = dist_vect**2
    total_dist_14 = np.sum(dist_2,axis = 2)**-7
    total_dist_14 = np.nan_to_num(total_dist_14, posinf= 0 )


    final = np.array([dist_vect[:,:,i]*total_dist_14 for i in range(3)])

    final = np.sum(final,axis = 1).transpose().flatten()

    return -12*final

def V_dd_dx(R):
    #R = np.array(np.split(x0,len(x0)/3)).astype(np.float)
    dist = R[np.newaxis, :, :] - R[:, np.newaxis, :]

    # p1 = (x_k-x_j)r_kj^(-5)
    dist_2 = dist**2
    total_dist_5 = np.nan_to_num((np.sum(dist_2,axis = 2))**(-5/2), posinf= 0 )

    p1 = np.sum(np.array([dist[:,:,i]*total_dist_5 for i in range(3)]),axis = 1).transpose().flatten()

    # p2 = 5*(x_k*x_j)*(e dot r_kj)^2*r_kj^-7 
    total_dist_7 = (np.sum(dist_2,axis = 2))**(-7/2)
    total_dist_7 = np.nan_to_num(total_dist_7, posinf= 0 )
    E_dot_R_k = e_i*R # e dot r_k:
    dist_er = (E_dot_R_k[np.newaxis, :, :] - E_dot_R_k[:, np.newaxis, :]) # e dot r_k - e dot r_j
    dist_er2 = (np.sum(dist_er,axis = 2))**2

    # print(total_dist_7)
    # print(dist_er2)
    new = total_dist_7*dist_er2
    p2 = np.array([dist[:,:,i]*new for i in range(3)])
    p2 = np.sum(p2,axis = 1).transpose().flatten()*-5
    # print(p2)

    y = (dist_er.transpose()*total_dist_5.transpose()).transpose()
    # print(y)

    x = np.sum(np.sum(y,axis = 1),axis = 1)
    p3 = -2*np.multiply.outer(x,e_i).flatten()
    #print(p3)
    V_dd_dx =  (p1+p2+p3)*(-(3*C_dd)/(4*np.pi))
    # print(V_dd_dx)
    return V_dd_dx

def V_trap_dx(x0):
    N = len(x0)/3
    W = np.array(trap_f_array_dx*int(N)).astype(np.float)
    V_trap_dx = W*x0
    #print(V_trap_dx)
    return V_trap_dx

def V_total_dx_array(x0):
    R = np.array(np.split(x0,len(x0)/3), dtype=float)
    V_dx_array = V_dd_dx(R)+V_rep_dx(R)+V_trap_dx(x0)
    return V_dx_array
