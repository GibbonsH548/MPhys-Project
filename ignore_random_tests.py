import numpy as np
import pandas as pd
import Energy_Derivative_New as EDF
# import EnergyDerivativeFunction as E_old

e = np.array([1,0,1])
x0 = np.array([0,1,2,3,4,5,10,11,12,-5,-7,-9])
R = np.array(np.split(x0,len(x0)/3), dtype=float)
dist_vect = R[np.newaxis, :, :] - R[:, np.newaxis, :]
# dist_2 = dist_vect**2
# c = np.sum(dist_2,axis = 2)
# np.fill_diagonal(c,1)
# print(c)
# print(np.reshape(R))
dist_2 = dist_vect**2
total_dist_14 = np.sum(dist_2,axis = 2)**-7
total_dist_14 = np.nan_to_num(total_dist_14, posinf= 0 )
#print(total_dist_14)

dist_new = dist_vect.transpose()
#print(dist_new)
print(np.sum(dist_new*total_dist_14,axis = 2).transpose().flatten())

final = np.array([dist_vect[:,:,i]*total_dist_14 for i in range(3)])
V_rep_dx = np.sum(final,axis = 1).transpose().flatten()
print(V_rep_dx)
#print(c)

# print(x0*e)
# EDF.V_dd_dx(R)
# print(R - R.reshape(R.shape[0], 1, 3))
# # p2 = -5 * (x_k - x_j)(e dot r_kj)^2 * r_kj^-5
# dist_vect = R[np.newaxis, :, :] - R[:, np.newaxis, :]
# dist_2 = dist_vect**2
# e_dot_dist = np.dot(dist_vect,e)
# e_dot_dist_2 = e_dot_dist**2
# total_dist_7 = np.nan_to_num((np.sum(dist_2,axis = 2))**(-7/2), posinf= 0 )
# ed2_td7 = e_dot_dist_2*total_dist_7
# p2 = -5*np.sum(np.array([dist_vect[:,:,i]*ed2_td7 for i in range(3)]),axis = 1).transpose().flatten()

# # p3 = 2e_k*(e dot r_kj)* r_kj^-5
# total_dist_5 = np.nan_to_num((np.sum(dist_2,axis = 2))**(-5/2), posinf= 0 )
# ed_td5 = e_dot_dist*total_dist_5
# p3 = np.sum(np.array([e[i]*ed_td5 for i in range(3)]),axis = 1).transpose().flatten()

# print(p3)
