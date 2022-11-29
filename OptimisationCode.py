import numpy as np
import pandas as pd
from scipy import optimize
import Energy_Derivative_New as EDF

import time

rng = np.random.default_rng() # can seed here if needed
for i in range(100):
    N = 50
    xy = rng.normal(loc = 0, scale = 30, size = 2*N)
    z = rng.normal(loc = 0, scale = 1, size = N)
    x0 = np.append(xy, z)
    #print(EDF.V_total(x0))

    #print(x0)#
    #start_time = time.time()
    res = optimize.minimize(EDF.V_total, x0, method="bfgs", jac=EDF.V_total_dx_array, options={"gtol": 1e-100})  # , jac = "3-point",options = {'gtol':1e-10000})#,jac = V_total_dx, options = {'gtol':1e-10000})#,jac = V_dx_array) #jac = V_dx_array
    #print("My program took", time.time() - start_time, "to run")
    e = EDF.V_total(res.x)
    #print(e)

    if i == 0:
        position_min_test = np.split(res.x,len(res.x)/3)

        Test = pd.DataFrame({"energy": [e], "positions": [res.x]})

    position_min_test = np.concatenate([position_min_test, np.split(res.x,len(res.x)/3)])
    Test.loc[i] = [e, res.x]
    print(i)

#print(Test["positions"][0][1])
Test.to_pickle("testdatatesting50_i.pkl")
