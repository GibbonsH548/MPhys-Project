#!/usr/bin/env python

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("Outputs/pd_outputs/DataframeFor21Atoms_50itterations.pkl")#"Outputs/testdatatesting50_i.pkl")
df_sorted = df.sort_values(by = "energy").reset_index()
df_sorted = df_sorted.drop("index",axis = 1)
print(df_sorted["positions"][1])

def PlotScatter(df_sorted, e_min_index, e_max_index, size):
    x1 = []
    y1 = []
    z1 = []

    for i in range(e_min_index,e_max_index):#(9879,9971):
        for j in range(len(df_sorted["positions"][0])):
            if j%3 ==0:            
                x1.append(df_sorted["positions"][i][j])
            if (j-1)%3 ==0:
                y1.append(df_sorted["positions"][i][j])
            if (j-2)%3 == 0:
                z1.append(df_sorted["positions"][i][j]) 
    plt.scatter(x1, y1, s = size)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plt.scatter(x1, z1, s = 1)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.show()
    plt.scatter(y1, z1, s = 1)
    plt.xlabel("y")
    plt.ylabel("z")
    plt.show()

def Plot_Energy_vs_Index(df_sorted, x_min, x_max):
    index = []
    energies = []
    for i in range(x_min,x_max):
        index.append(i)
        energies.append(df_sorted["energy"][i])#*kg**(-1)*(t**2))
        #print(df_sorted["energy"][i])

    plt.scatter(index,energies, s = 1)
    plt.xlabel("index (ordered)")
    plt.ylabel("Energy (rescaled)")
    #plt.show()

def Plot_Energy_vs_Index_n(df_sorted, x_min, x_max):
    index = []
    energies = []
    for i in range(x_min,x_max):
        index.append(df_sorted["index"][i])
        energies.append(df_sorted["energy"][i])#*kg**(-1)*(t**2))
        #print(df_sorted["energy"][i])

    plt.scatter(index,energies, s = 1)
    plt.xlabel("index (ordered)")
    plt.ylabel("Energy (rescaled)")
    #plt.show()

def SaveBasins(name, df_sorted):
    df_new = df_sorted.round(decimals = 10)
    N_basins = df_new["energy"].nunique()
    df_reduced = df_new.drop_duplicates(subset=["energy"], keep = "first").reset_index()
    df_red = df_reduced.drop(columns = ["index","energy"])
    d = df_red.to_numpy()
    d = np.insert(d,0,N_basins)
    np.savetxt(f"Outputs/{name}.txt", d, fmt = "%s")

# SaveBasins("20_atoms", df_sorted)
Plot_Energy_vs_Index(df_sorted, 0, 49)
plt.show()
# ot_Energy_vs_Index(df_sorted, 0, 170)
# plt.show()
# PlotScatter(df_sorted, 0, 19, 1)
# PlotScatter(df_sorted, 197, 198, 1)
# PlotScatter(df_sorted, 198, 199, 1)
#PlotScatter(df_sorted, 0, 950, 0.1)
#PlotScatter(df_sorted, 0, 100, 0.1)
PlotScatter(df_sorted, 0, 100, 0.5)
# PlotScatter(df_sorted, 800, 900, 0.5)
# PlotScatter(df_sorted, 998, 999, 5)
plt.show()
