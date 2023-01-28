import numpy as np
import pandas as pd
from scipy import optimize
import Energy_Derivative_New as EDF
import tomli

with open("input.toml", "rb") as f:
    input = tomli.load(f)

N = input["simulation_properties"]["particles"]   
it = input["simulation_properties"]["minimization_iterations"]   
txt_output = input["simulation_properties"]["output_basin_txt_file"]   
pd_output = input["simulation_properties"]["output_pd_energy_and_coordinates_file"]
gs_output = input["simulation_properties"]["ground_state_coordinates_for_QMC"]

rng = np.random.default_rng() # can seed here if needed
for i in range(it):
    X = rng.normal(loc = 0, scale = 10, size = N) # N positions of coordinates in x cartesian directions
    Y = rng.normal(loc = 0, scale = 10, size = N) # N positions of coordinates in y cartesian directions
    Z = rng.normal(loc = 0, scale = 1, size = N) # N positions of coordinates in z cartesian directions
    x0 = np.array([X,Y,Z]).transpose().flatten() # setting it to [x1, y1, z1,... , xN, yN, zN]

    res = optimize.minimize(EDF.V_total, x0, method="bfgs", jac=EDF.V_total_dx_array, options = {'gtol':1e-100000})  # , jac = "3-point" - numerical approximation
    e = EDF.V_total(res.x)

    if i == 0:
        position_min_test = np.split(res.x,len(res.x)/3)
        Test = pd.DataFrame({"energy": [e], "positions": [res.x]})

    position_min_test = np.concatenate([position_min_test, np.split(res.x,len(res.x)/3)])
    Test.loc[i] = [e, res.x]
    #print(i)

if pd_output == "T":
    name = "Outputs/pd_outputs/DataframeFor{}Atoms_{}itterations".format(N,it)
    filename = "%s.pkl" % name
    Test.to_pickle(filename)


if txt_output == "T":
    df = Test
    df_sorted = df.sort_values(by = "energy").reset_index()
    df_sorted = df_sorted.drop("index",axis = 1)
    df_sorted['energy'] = df_sorted['energy'].round(6)


    df_new = df_sorted.drop_duplicates(subset=['energy'], keep='first')
    array = df_new["positions"].to_numpy()

    for i in range(len(array)):
        a = np.array(np.split(array[i],len(array[i])/3))
        if i == 0 :
            thearray = a
        else: 
            thearray = np.vstack([thearray,a])

    name = "Outputs/BasinLocations/CoordinatesOf{}AtomsinBasins".format(N)
    filename = "%s.txt" % name
    np.savetxt(filename,thearray, header=str(len(array)), comments="")

if gs_output == "T":
    df = Test
    df_sorted = df.sort_values(by = "energy").reset_index()
    df_sorted = df_sorted.drop("index",axis = 1)
    array = df_sorted["positions"][0]
    ones = np.ones(len(array),dtype = object)[:,None]
    array2 = array.reshape(len(array),1)
    # print(array2)

    thearray = np.concatenate((array2, ones),axis=1)#,dtype=object)

    #print(thearray)
    #print(gs_output)

    name = "Outputs/GroundStateCoordinatesforQMC/CoordinatesOfGroundState{}".format(N)
    filename = "%s.txt" % name
    np.savetxt(filename,thearray,fmt= ['%1.16E','%d'], comments="")
    # print(df_sorted)
    # print(df_sorted["positions"][0])



