#Equations and functions for the moon station keeping problem with RL
#Functions: running_mean, choose_Halo, data_Halos, rv_Halo

import numpy as np
from random import randint
from numpy.linalg import norm
from environment.CR3BP import *
from environment.rk4 import *


#System physicial constants
solar_day = 86400  #solar day [s]
mu = 0.01215059  #mass ratio []
l_star = 3.844e+5  #system characteristic length [km]
t_star = 3.751903e+5  #system characteristic time [s]
v_star = l_star/t_star  #system characteristic velocity [km/s]
g0 = 9.80665e-3  #sea-level gravitational acceleration [km/s^2]



#Calculates the mean of the state at every step
def running_mean(mean, step, new_value):  
        
    new_mean = 1./float(step)*((float(step)-1.)*mean+new_value)

    return new_mean


#Chooses randomly a set of initial conditions (r0, v0, C, T) for a L1 Halo orbit to be propagated afterwards   
def data_Halos(filename, num_steps, num_Halo, first_trajectory, num_halos_skip):  #filename = l1_halo_north

    r0_Halo_all = []
    v0_Halo_all = []
    T_Halo_all = []
    C_Halo_all = []

    with open(filename, 'r') as file:
        #file.readline()
        file_all = file.readlines()

        for i in range(num_Halo):  #read line
            line = file_all[first_trajectory + i*num_halos_skip]
            line = line.split()
            state = np.array(line).astype(np.float64)  #non-dimensional data
            
            #save initial conditions
            r0_Halo_all.append(np.array([state[0],state[1],state[2]]))
            v0_Halo_all.append(np.array([state[3],state[4],state[5]]))
            T_Halo_all.append(state[6])
            C_Halo_all.append(state[10])
    
    r_Halo_all = []
    v_Halo_all = []

    for i in range(len(r0_Halo_all)):
        r_Halo_i, v_Halo_i = rv_Halo(r0_Halo_all[i], v0_Halo_all[i], 0., T_Halo_all[i], num_steps)
        r_Halo_all.append(r_Halo_i)
        v_Halo_all.append(v_Halo_i)

    return r_Halo_all, v_Halo_all, T_Halo_all, C_Halo_all



#Obtains r_Halo and v_Halo vectors of the reference Halo orbit
def rv_Halo(r0, v0, t0, tf, num_steps):

    t_eval = np.linspace(t0, tf, num_steps+1) 
    y0 = np.concatenate((r0,v0), axis=None)

    r_Halo = [r0]
    v_Halo = [v0]
    data = ()
    
    for i in range(len(t_eval)-1):
        t_span = [t_eval[i], t_eval[i+1]] 
        sol = rk4_prec(CR3BP_equations_free, y0, t_span[0], t_span[1], 1e-7, data)
        r_Halo_new = np.array([sol[0],sol[1],sol[2]])
        v_Halo_new = np.array([sol[3],sol[4],sol[5]])
        r_Halo.append(r_Halo_new)
        v_Halo.append(v_Halo_new)
        y0 = sol

    return r_Halo, v_Halo
