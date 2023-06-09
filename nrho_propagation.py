#NRHO orbit propagation from IC

import os
import sys
import time
import numpy as np
from numpy.linalg import norm
from environment.CR3BP import propagate_cr3bp_free, Jacobi_const, CR3BP_eqs_free_lsoda
from scipy.integrate import solve_ivp


#Initial conditions (wrt to L2 point)
mu = 0.012150529811497 

x0 = -0.167809008979642
y0 = 0.
z0 = 0.0286160249739854

vx0 = 0.
vy0 = 0.882937858803859
vz0 = 0.

#Period of the orbit
Dt = 2.030900008526994

#L2 position
X_L2 = 1.1556




def CR3BP_equations_free(t, state, mu):  #without control
    '''
    input: -t (time)
           -state
    
    output: -state_dot (derivative of the state)

    '''

    #state variables (state=[x y z vx vy vz])
    state_dot = np.array(state.shape)

    #spacecraft position
    x=state[0]
    y=state[1]
    z=state[2]

    #spacecraft velocity
    vx=state[3]
    vy=state[4]
    vz=state[5]

    #additional quantities
    r13=np.sqrt((x+mu)**2+y**2+z**2)  #earth-sc distance
    r23=np.sqrt((x-(1-mu))**2+y**2+z**2)  #moon-sc distance

    #obtain derivative of the state
    state_dot_x=vx
    state_dot_y=vy
    state_dot_z=vz

    state_dot_vx=2*vy+x-((1-mu)*(x+mu))/(r13**3)-(mu*(x-1+mu))/(r23**3)
    state_dot_vy=-2*vx+y-((1-mu)*y)/(r13**3)-(mu*y)/(r23**3)
    state_dot_vz=z*((-(1-mu)/(r13**3))+(-mu/(r23**3)))


    state_dot = [state_dot_x, state_dot_y, state_dot_z, state_dot_vx, state_dot_vy, state_dot_vz]

    return state_dot





#Create file txt with NRHO orbit
f_out_L2 = open('NRHO_L2_rv.txt', "w")
f_out_L2.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("x", "y", "z", "vx", "vy", "vz", "C", "T"))

#How many points for the propagation
N_times = 100

#L2 orbit propagation
t_span = [0., Dt]
t_eval = np.linspace(0., Dt, N_times)
r0 = np.array([x0+X_L2, y0, z0])  #x0 is wrt to L2 
v0 = np.array([vx0, vy0, vz0])

s0 = np.concatenate((r0, v0), axis=None)  #initial state
sol = solve_ivp(fun=CR3BP_equations_free, t_span=t_span, y0=s0, method='RK45', t_eval=t_eval, args=(mu,))
    
for j in range(N_times):
    C = Jacobi_const(sol.y[0][j], sol.y[1][j], sol.y[2][j], sol.y[3][j], sol.y[4][j], sol.y[5][j])
    f_out_L2.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
            % (sol.y[0][j], sol.y[1][j], sol.y[2][j], sol.y[3][j], sol.y[4][j], sol.y[5][j], C, Dt))
f_out_L2.write("\n\n")
f_out_L2.close()
