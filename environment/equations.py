#Equations and functions for the moon station keeping problem with RL

#functions: rk4, running_mean, choose_Halo, rv_Halo

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


#Runge-kutta integration method
def rk4(f, y0, t_eval, data):

    n_eq=len(y0)
    t0=t_eval[0]
    dt=t_eval[1]-t_eval[0]

    f0=np.zeros(n_eq)
    f(t0, y0, f0, data)
    
    t1=t0+dt/2.0
    y1=y0+dt*f0/2.0
    f1=np.zeros(n_eq)
    f(t1, y1, f1, data)
    
    t2=t0+dt/2.0
    y2=y0+dt*f1/2.0
    f2=np.zeros(n_eq)
    f(t2, y2, f2, data)
    
    t3=t0+dt	
    y3=y0+dt*f2
    f3=np.zeros(n_eq)
    f(t3, y3, f3, data)
    
    y=y0+dt*(f0+2.0*f1+2.0*f2+f3)/6.0

    return y


#calculates the mean of the state at every step
def running_mean(self, mean, step, new_value):  
        
    new_mean=1./float(step)*((float(step)-1.)*mean+new_value)

    return new_mean
    


#chooses randomly a set of initial conditions (r0,v0,T) for a L1 Halo orbit to be propagated afterwards   
def choose_Halo(filename, single_matrix):
    with open(filename, 'r') as f:
        lines=f.readlines()
        rv_matrix=[]
        for line in lines:
            rv_matrix.append(line.split(' '))

    k=randint(2,101)
    i=randint(1,245)  #select a random matrix 
    if single_matrix:  #extract only from the first matrix (Halo-1)
        r0=np.array(rv_matrix[k][0:3])  #initial position
        v0=np.array(rv_matrix[k][3:6])  #initial velocity

    else:  #extract from any of the matrices (Halo-j)
        j=102*i+k
        r0=np.array(rv_matrix[j][0:3])  #initial position
        v0=np.array(rv_matrix[j][3:6])  #initial velocity

    return r0, v0
    



#obtain r_Halo and v_Halo vectors of the reference Halo orbit
def rv_Halo(r0,v0, t0, tf, num_steps):

    t_eval=np.linspace(t0, tf, num_steps) 
    y0=np.concatenate((r0,v0),axis=None)

    r_Halo=[r0]
    v_Halo=[v0]
    data=()
    
    for i in len(t_eval-1):
        t_span=[t_eval[i],t_eval[i+1]]
        sol=rk4_prec(CR3BP_equations_free, y0, t_span[0], t_span[1], 1e-7, data)
        r_Halo_new=np.array([sol[0],sol[1],sol[2]])
        v_Halo_new=np.array([sol[3],sol[4],sol[5]])
        r_Halo.append(r_Halo_new)
        v_Halo.append(v_Halo_new)
        y0=sol

    return r_Halo, v_Halo