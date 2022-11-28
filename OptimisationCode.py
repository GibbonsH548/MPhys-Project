import numpy as np
import pandas as pd
import scipy.optimize 
import Energy_Derivative_New as EDF

import warnings
import time

#suppress warnings
warnings.filterwarnings('ignore')
#np.random.seed(10)
a = 0
Test = pd.DataFrame(columns = ["energy","positions"])
for i in range (100):
    N = 50
    x = np.array(np.random.normal(loc = 0, scale = 30, size = N).astype(np.float)) # gaussian
    y = np.array(np.random.normal(loc = 0, scale = 30, size = N).astype(np.float)) # gaussian
    z = np.array(np.random.normal(loc = 0, scale = 1, size = N).astype(np.float)) # gaussian
    x0 = np.concatenate((x,y,z)).transpose().flatten()
    #print(EDF.V_total(x0))

    #print(x0)#
    #start_time = time.time()
    res = scipy.optimize.minimize(EDF.V_total, x0, method='bfgs', jac = EDF.V_total_dx_array ,options = {'gtol':1e-100})#, jac = "3-point",options = {'gtol':1e-10000})#,jac = V_total_dx, options = {'gtol':1e-10000})#,jac = V_dx_array) #jac = V_dx_array
    #print("My program took", time.time() - start_time, "to run")
    e = EDF.V_total(res.x)
    #print(e)

    pd.DataFrame()
    if a == 0:
        position_min_test = np.split(res.x,len(res.x)/3)
        a += 1
        
        s = pd.Series([e,res.x], index=Test.columns)
        Test = Test.append(s,ignore_index=True)
    if a == 1:
        position_min_test = np.concatenate([position_min_test, np.split(res.x,len(res.x)/3)])
        s = pd.Series([e,res.x], index=Test.columns)
        Test = Test.append(s,ignore_index=True)
        print(i)

#print(Test["positions"][0][1])
Test.to_pickle("C:/Users/gibbo/OneDrive - Lancaster University/Uni stuff/Year 4/MPhys Project/Project Code/testdatatesting50_i.pkl")
