#Equations and functions for the moon station keeping problem with RL

#functions: running_mean, choose_Halo, rv_Halo

import numpy as np
from random import randint
from numpy.linalg import norm
from environment.CR3BP import *
from environment.rk4 import *


#System constants
solar_day=86400 #solar day [s]
mu=0.01215059 #mass ratio []
l_star=3.844e+5 #system characteristic length [km]
t_star=3.751903e+5 #system characteristic time [s]
v_star=l_star/t_star #system characteristic velocity [km/s]
g0=9.80665e-3 #sea-level gravitational acceleration [km/s^2]



#calculates the mean of the state at every step
def running_mean(mean, step, new_value):  
        
    new_mean=1./float(step)*((float(step)-1.)*mean+new_value)

    return new_mean
    


#chooses randomly a set of initial conditions (r0,v0,T) for a L1 Halo orbit to be propagated afterwards   
def choose_Halo(filename, single_matrix):
    with open(filename, 'r') as f:
        line=f.readline()
        lines=f.readlines()
        rv_matrix=[]
        for line in lines:
            line_split = line.split()
            if len(line_split) > 0:
                vec = np.array(line_split).astype(np.float64)
                rv_matrix.append(vec)
    
    k=randint(0,100)
    i=randint(0,247)  #select a random matrix 
    if single_matrix:  #extract only from the first matrix (Halo-1)
        r0=np.array(rv_matrix[k][0:3])  #initial position
        v0=np.array(rv_matrix[k][3:6])  #initial velocity
        T_Halo = rv_matrix[k][7]
        C_Halo = rv_matrix[k][6]

    else:  #extract from any of the matrices (Halo-j)
        j=101*i+k
        r0=np.array(rv_matrix[j][0:3])  #initial position
        v0=np.array(rv_matrix[j][3:6])  #initial velocity
        T_Halo = rv_matrix[j][7]
        C_Halo = rv_matrix[j][6]
     

    return r0, v0, T_Halo, C_Halo
    



#obtain r_Halo and v_Halo vectors of the reference Halo orbit
def rv_Halo(r0, v0, t0, tf, num_steps):

    t_eval=np.linspace(t0, tf, num_steps+1) 
    y0=np.concatenate((r0,v0), axis=None)

    r_Halo=[r0]
    v_Halo=[v0]
    data=()
    
    for i in range(len(t_eval)-1):
        t_span=[t_eval[i],t_eval[i+1]] 
        sol=rk4_prec(CR3BP_equations_free, y0, t_span[0], t_span[1], 1e-7, data)
        r_Halo_new=np.array([sol[0],sol[1],sol[2]])
        v_Halo_new=np.array([sol[3],sol[4],sol[5]])
        r_Halo.append(r_Halo_new)
        v_Halo.append(v_Halo_new)
        y0=sol

    return r_Halo, v_Halo

