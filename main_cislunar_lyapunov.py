import os
import sys
import time
import numpy as np
import math
from numpy.linalg import norm
from environment.CR3BP import propagate_cr3bp_free, Jacobi_const

#Read PO files
x_lyp_L1 = [] #Lyapunov orbits around L1: x
y_lyp_L1 = [] #Lyapunov orbits around L1: y
z_lyp_L1 = [] #Lyapunov orbits around L1: z
vx_lyp_L1 = [] #Lyapunov orbits around L1: vx
vy_lyp_L1 = [] #Lyapunov orbits around L1: vy
vz_lyp_L1 = [] #Lyapunov orbits around L1: vz
T_lyp_L1 = [] #Lyapunov orbits around L1: T

x_lyp_L2 = [] #Lyapunov orbits around L2: x
y_lyp_L2 = [] #Lyapunov orbits around L2: y
z_lyp_L2 = [] #Lyapunov orbits around L2: z
vx_lyp_L2 = [] #Lyapunov orbits around L2: vx
vy_lyp_L2 = [] #Lyapunov orbits around L2: vy
vz_lyp_L2 = [] #Lyapunov orbits around L2: vz
T_lyp_L2 = [] #Lyapunov orbits around L2: T

PO_folder = "PO_files/"
PO_name_L1 = "L1_lyapunov.dat"
PO_name_L2 = "L2_lyapunov.dat"

PO_file_L1 = PO_folder + PO_name_L1 #File with PO data
with open(PO_file_L1, "r") as f_L1: # with open context
    f_L1.readline()
    file_all = f_L1.readlines()
    for line in file_all: #read line
        line = line.split()
        state = np.array(line).astype(np.float64) #non-dimensional data
        
        #save data
        x_lyp_L1.append(state[0])
        y_lyp_L1.append(state[1])
        z_lyp_L1.append(state[2])
        vx_lyp_L1.append(state[3])
        vy_lyp_L1.append(state[4])
        vz_lyp_L1.append(state[5])
        T_lyp_L1.append(state[6])
f_L1.close() #close file

PO_file_L2 = PO_folder + PO_name_L2 #File with PO data
with open(PO_file_L2, "r") as f_L2: # with open context
    f_L2.readline()
    file_all = f_L2.readlines()
    for line in file_all: #read line
        line = line.split()
        state = np.array(line).astype(np.float64) #non-dimensional data
        
        #save data
        x_lyp_L2.append(state[0])
        y_lyp_L2.append(state[1])
        z_lyp_L2.append(state[2])
        vx_lyp_L2.append(state[3])
        vy_lyp_L2.append(state[4])
        vz_lyp_L2.append(state[5])
        T_lyp_L2.append(state[6])
f_L2.close() #close file

#Create file with Lyapunov orbits
f_out_L1 = open(PO_folder + "L1_lyap_orbits.txt", "w") # open file
f_out_L2 = open(PO_folder + "L2_lyap_orbits.txt", "w") # open file
f_out_L1.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("# x", "y", "z", "vx", "vy", "vz", "C"))
f_out_L2.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("# x", "y", "z", "vx", "vy", "vz", "C"))

#Orbital propagation
N_orbits = len(x_lyp_L1)
N_times = 200

#L1 orbits
for i in range(N_orbits):
    t_span = [0., T_lyp_L1[i]]
    t_eval = np.linspace(0., T_lyp_L1[i], N_times)
    r0 = np.array([x_lyp_L1[i], y_lyp_L1[i], z_lyp_L1[i]])
    v0 = np.array([vx_lyp_L1[i], vy_lyp_L1[i], vz_lyp_L1[i]])

    r_lyp_L1, v_lyp_L1 = \
        propagate_cr3bp_free(r0, v0, \
            t_span, t_eval = t_eval)
    
    for j in range(N_times):
        C = Jacobi_const(r_lyp_L1[j][0], r_lyp_L1[j][1], r_lyp_L1[j][2], v_lyp_L1[j][0], v_lyp_L1[j][1], v_lyp_L1[j][2])
        f_out_L1.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
                % (r_lyp_L1[j][0], r_lyp_L1[j][1], r_lyp_L1[j][2], v_lyp_L1[j][0], v_lyp_L1[j][1], v_lyp_L1[j][2], C))
    f_out_L1.write("\n\n")
f_out_L1.close()

#L2 orbits
for i in range(N_orbits):
    t_span = [0., T_lyp_L2[i]]
    t_eval = np.linspace(0., T_lyp_L2[i], N_times)
    r0 = np.array([x_lyp_L2[i], y_lyp_L2[i], z_lyp_L2[i]])
    v0 = np.array([vx_lyp_L2[i], vy_lyp_L2[i], vz_lyp_L2[i]])

    r_lyp_L2, v_lyp_L2 = \
        propagate_cr3bp_free(r0, v0, \
            t_span, t_eval = t_eval)
    
    for j in range(N_times):
        C = Jacobi_const(r_lyp_L2[j][0], r_lyp_L2[j][1], r_lyp_L2[j][2], v_lyp_L2[j][0], v_lyp_L2[j][1], v_lyp_L2[j][2])
        f_out_L2.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
                % (r_lyp_L2[j][0], r_lyp_L2[j][1], r_lyp_L2[j][2], v_lyp_L2[j][0], v_lyp_L2[j][1], v_lyp_L2[j][2], C))
    f_out_L2.write("\n\n")
f_out_L2.close()

