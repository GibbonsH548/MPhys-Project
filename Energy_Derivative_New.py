import numpy as np
from numba import njit
from scipy.spatial import distance as sd

import warnings
#suppress warnings (to stop printing divide by 0 error)
#warnings.filterwarnings('ignore')

# hbar, M - atom, 1/w

# Constants:
# Real values 
m_i = 2.72328406922e-25  # Mass of particles (kg)

mu_0_i = 1.257*10**(-7)  # Vacuum permability ()
mu_b_i = 9.27400968e-24  # Bohr magneton ()
mu_m = 10*mu_b_i         # magnetic moment of particles

e_i = np.array([0,0,1], dtype=np.float64)  # Unit vector of dipole orientation

# trapping frequency: If w_p << w_z - pancake if w_p >> w_z - cigar
w_p_i = 1               # axial trapping frequency
w_z_i = 50          # radial trapping frequency
# Too big problem with derivatives?

# Converting to sensible units 
kg = m_i**(-1)
m = m_i*kg     # convert to mass = 1

w_2_i = 2*w_p_i**2 + w_z_i**2   # converting to w**2 = 1

Hz = (w_2_i**(-1/2))        # Converting Hz 
w_p = w_p_i*Hz              # converting axial trapping frequency
w_z = w_z_i*Hz              # converrting radial trapping frequency

mu_0 = mu_0_i*kg*Hz**(2)    # converting vacuum permabilty to new units


C_dd = mu_0*(mu_m)**2    # C_dd is the the coupling constant  - μ_0 μ_m^2 


trap_f_array = np.array([(w_p**2)*(m/2),(w_p**2)*(m/2),(w_z**2)*(m/2)])
trap_f_array_dx = np.array([(w_p**2)*(m),(w_p**2)*(m),(w_z**2)*(m)])


@njit
def V_trap(R):   
    """ Calculates the trapping potential of the system
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    """
    V_trap = np.dot(trap_f_array,np.sum(np.square(R),axis = 0))    # dot product of trapping frequencies and sum of all x^2,y y^2, z^2
    return V_trap

@njit
def V_dd(R, distance_array):
    """ Calculates the potential energy of the system due to the dipole interactions between each particle
    Parameters
    ----------
        R: 2D numpy array - shape (N,3)
    """
    dist_vect = R - R.reshape(R.shape[0], 1, 3) # Subtracts all particle postions from all others gives displacement vectors in a np array (np.newaxis increases the dimension of the array from 2 -> 3) 

    V_dd_1 = np.sum(distance_array**-3) # Calulates the sum of all rij^-3
    e_dot_rij_sqr_wz = np.tril(np.sum(dist_vect*e_i,axis=2)**2).flatten() # Produces an array of the dot products or the dipolar unit vector with the diplacement between particles 
    e_dot_rij_sqr = e_dot_rij_sqr_wz[e_dot_rij_sqr_wz!=0] # Removes the zeros from duplicates and rij where i = j (ie)

    V_dd_2 = np.dot(distance_array**-5,e_dot_rij_sqr) # Calculates the dot product of rij^-5 with (e dot rij)^2 which finds the total potential from all particles 
    V_dd = (C_dd/(4*np.pi))*(V_dd_1-3*V_dd_2) 
    return V_dd

@njit
def V_repulsive(distance_array):
    """ Calculates the potential energy of the system due to a r_ij^12 repulsive potential between each particle
    Parameters
    ----------
        R: 2D numpy array - shape (N,3)
    """
    sp_result = distance_array**(-12) # Array of distances between particle^-12 (for 3 particles [r12^-12,r13^-12,r23^-12])
    V_rep = np.sum(sp_result)
    return V_rep


def V_total(x0):
    """ Calculates the total potential energy of a system of dipolar particles
    Parameters  
    ----------
        x0: 1D numpy array - shape (N*3)  
                           - [x1, y1, z1, ..., xN, yN, zN]
    """
    R = np.reshape(x0, (x0.shape[0] // 3, 3))  # Splitting x0 into a 2D np array - [[x1, y1, z1],...,[xN, yN, zN]]
    distance_array = sd.pdist(R)               # Array of distances between particles (for 3 particles [r12,r13,r23])
    V_tot = V_trap(R)+V_repulsive(distance_array)+V_dd(R, distance_array)   # Suming over all contributions to the overall potential energy
    return V_tot


# Derivative Array:

@njit
def V_rep_dx(dist_vect, dist_2):
    """ Returns an array of the derivatives of the repulsive potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_rep])
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    """

    # array of distances between particles 
    # [[[x00,y00,z00],[x01,y01,z01]],[[x10,y10,z10],[x11,y11,z11]]]

    dist_14 = np.sum(dist_2,axis = 2)

    np.fill_diagonal(dist_14, 1)
    total_dist_14=dist_14**-7
    np.fill_diagonal(total_dist_14, 0)

    final = dist_vect.transpose()*total_dist_14
    V_rep_dx = np.sum(final,axis = 2).transpose().flatten()
    return -12*V_rep_dx

@njit
def V_dd_dx(dist_vect, dist_2):
    """ Returns an array of the derivatives of the dipolar interaction potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_rep])
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    """

    # p1 = (x_k-x_j)r_kj^(-5)
    total_dist_5 = np.sum(dist_2,axis = 2)**(-5/2)
    np.fill_diagonal(total_dist_5, 0)
    p1 = np.sum(dist_vect.transpose()*total_dist_5,axis = 2).flatten()
    
    # p2 = -5 * (x_k - x_j)(e dot r_kj)^2 * r_kj^-7
    e_dot_dist = np.sum(dist_vect*e_i,axis=2)
    e_dot_dist_2 = e_dot_dist**2
    total_dist_7 = np.sum(dist_2,axis = 2)**(-7/2)
    np.fill_diagonal(total_dist_7,0)
    ed2_td7 = e_dot_dist_2*total_dist_7
    p2 = -5*np.sum(dist_vect.transpose()*ed2_td7,axis = 2).transpose().flatten()

    # p3 = 2e_k*(e dot r_kj)* r_kj^-5
    total_dist_5 = np.sum(dist_2,axis = 2)**(-5/2)
    np.fill_diagonal(total_dist_5,0)
    ed_td5 = e_dot_dist*total_dist_5

    C = np.repeat(e_i,ed_td5.shape[0]**2).reshape(3,ed_td5.shape[0],ed_td5.shape[0])*ed_td5

    p3 = 2*np.sum(C,axis = 1).transpose().flatten()

    V_dd_dx =  (p1+p2+p3)*(-(3*C_dd)/(4*np.pi))
    return V_dd_dx

@njit
def V_trap_dx(R):
    """ Returns an array of the derivatives of the trapping potential energy term for the system with respect to each parameter (ie [d/dx1,d/dy1,d/dz1,...,d/dxN,d/dyN,d/dzN]V_rep])
    Parameters  
    ----------
        R: 2D numpy array - shape (N,3)
    """
    V_trap_dx = (trap_f_array_dx*R).flatten()
    return V_trap_dx

@njit
def V_total_dx_array(x0):
    R = np.reshape(x0, (x0.shape[0] // 3, 3))
    dist_vect = R - R.reshape(R.shape[0], 1, 3) # Subtracts all particle postions from all others gives displacement vectors in a np array (np.newaxis increases the dimension of the array from 2 -> 3) 
    dist_2 = dist_vect**2 # square of distances 3D
    V_dx_array = V_dd_dx(dist_vect, dist_2)+V_rep_dx(dist_vect, dist_2)+V_trap_dx(R)
    return V_dx_array
