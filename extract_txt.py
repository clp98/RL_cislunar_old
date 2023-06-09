#Read txt file, extract initial conditions and propagate them to get Halo orbits

import os
import sys
import time
import numpy as np
from numpy.linalg import norm
from environment.CR3BP import propagate_cr3bp_free, Jacobi_const

#HALO ORBIT
#initialize vectors
x_Halo_L1=[]
y_Halo_L1=[]
z_Halo_L1=[]
vx_Halo_L1=[]
vy_Halo_L1=[]
vz_Halo_L1=[]
T_Halo_L1=[]

# x_Halo_L2=[]
# y_Halo_L2=[]
# z_Halo_L2=[]
# vx_Halo_L2=[]
# vy_Halo_L2=[]
# vz_Halo_L2=[]
# T_Halo_L2=[]


#read L1 Halo data
with open('PO_files/l1_halo_north.txt', "r") as file_L1: 
    file_L1.readline()
    file_all=file_L1.readlines()
    for line in file_all: #read line
        line=line.split()
        state=np.array(line).astype(np.float64) #non-dimensional data
        
        #save data
        x_Halo_L1.append(state[0])
        y_Halo_L1.append(state[1])
        z_Halo_L1.append(state[2])
        vx_Halo_L1.append(state[3])
        vy_Halo_L1.append(state[4])
        vz_Halo_L1.append(state[5])
        T_Halo_L1.append(state[6])

file_L1.close()

#Create file txt with Halo orbits
f_out_L1=open('Halo_L1_rv.txt', "w")
f_out_L1.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("x", "y", "z", "vx", "vy", "vz", "C", "T"))


N_orbits=len(x_Halo_L1)-1
N_times=101

#L1 orbit propagation
for i in range(N_orbits):
    t_span=[0., T_Halo_L1[i]]
    t_eval=np.linspace(0., T_Halo_L1[i], N_times)
    r0=np.array([x_Halo_L1[i], y_Halo_L1[i], z_Halo_L1[i]])
    v0=np.array([vx_Halo_L1[i], vy_Halo_L1[i], vz_Halo_L1[i]])

    s0=np.concatenate((r0, v0), axis=None)
    sol=propagate_cr3bp_free(s0, t_eval=t_eval)
    
    for j in range(N_times):
        C=Jacobi_const(sol[j][0], sol[j][1], sol[j][2], sol[j][3], sol[j][4], sol[j][5])
        f_out_L1.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
                % (sol[j][0], sol[j][1], sol[j][2], sol[j][3], sol[j][4], sol[j][5], C, T_Halo_L1[i]))
    f_out_L1.write("\n\n")
f_out_L1.close()
