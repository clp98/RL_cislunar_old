#Read txt file, extract initial conditions and propagate to get Halo orbit
#(this script will be use only once in oprder to get the Halo orbit initial condition)

import os
import sys
import time
import numpy as np
from numpy.linalg import norm
from environment.CR3BP import propagate_cr3bp_free, Jacobi_const

#initialize vectors
x_Halo_L1=[]
y_Halo_L1=[]
z_Halo_L1=[]
vx_Halo_L1=[]
vy_Halo_L1=[]
vz_Halo_L1=[]
T_Halo_L1=[]

x_Halo_L2=[]
y_Halo_L2=[]
z_Halo_L2=[]
vx_Halo_L2=[]
vy_Halo_L2=[]
vz_Halo_L2=[]
T_Halo_L2=[]


#read L1 Halo data
with open('L1_Halo_north.txt', "r") as file_L1: 
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
f_out_L1=open('Halo_L1_rv', "w")
f_out_L2=open('Halo_L2_rv', "w")
f_out_L1.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("x", "y", "z", "vx", "vy", "vz", "C"))
f_out_L2.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("x", "y", "z", "vx", "vy", "vz", "C"))


##Orbital Propagation
N_orbits=len(x_Halo_L1)
N_times=100

#L1 orbit propagation
for i in range(N_orbits):
    t_span=[0., T_Halo_L1[i]]
    t_eval=np.linspace(0., T_Halo_L1[i], N_times)
    r0=np.array([x_Halo_L1[i], y_Halo_L1[i], z_Halo_L1[i]])
    v0=np.array([vx_Halo_L1[i], vy_Halo_L1[i], vz_Halo_L1[i]])

    r_Halo_L1, v_Halo_L1=propagate_cr3bp_free(r0, v0, t_span, t_eval=t_eval)
    
    for j in range(N_times):
        C=Jacobi_const(r_Halo_L1[j][0], r_Halo_L1[j][1], r_Halo_L1[j][2], v_Halo_L1[j][0], v_Halo_L1[j][1], v_Halo_L1[j][2])
        f_out_L1.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
                % (r_Halo_L1[j][0], r_Halo_L1[j][1], r_Halo_L1[j][2], v_Halo_L1[j][0], v_Halo_L1[j][1], v_Halo_L1[j][2], C))
    f_out_L1.write("\n\n")
f_out_L1.close()
