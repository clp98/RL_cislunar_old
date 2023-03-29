#Equations and functions for the moon station keeping problem with RL

import numpy as np
from random import randint
from numpy.linalg import norm


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




def CR3BP(t, s, f, t_1, t_2, ueq): #with control
    """
    Right-hand side of the system of equations of motion of a spacecraft
    in the Earth-Moon circular restricted three body problem (CR3BP),
    with low thrust terms.
    The (nondimensional) state components are propagated with respect to the system barycenter in the
    Earth-Moon rotating reference frame.

    Args:
        t - (float) time (nondim)
        s - (np.array) spacecraft state (position, velocity, mass) (nondim) - 7 components (3+3+1)
        f - (np.array) spacecraft thrust (nondim) - 3 components
        t_1 - (float) start of thrusting period (nondim)
        t_2 - (float) end of thrusting period (nondim)
        ueq - (float) spacecraft equivalent ejection velocity (nondim)

    Return:
        s_dot - (np.array) derivatives of the spacecraft state (nondim)
    """

    #State variables

    #spacecraft position
    x=s[0]
    y=s[1]
    z=s[2]

    #spacecraft velocity
    vx=s[3]
    vy=s[4]
    vz=s[5]

    #spacecraft mass
    m=s[6]

    #Control
    if t>=t_1 and t<=t_2:
        fx=f[0]
        fy=f[1]
        fz=f[2]
        f_norm=norm(f)
    else:
        fx=0.
        fy=0.
        fz=0.
        f_norm=0.

    #Auxiliary variables
    r13=sqrt((x+mu)**2+y**2+z**2) #Earth-S/C distance
    r23=sqrt((x-1.+mu)**2+y**2+z**2) #Moon-S/C distance

    #Equations of motion
    x_dot=vx
    y_dot=vy
    z_dot=vz

    vx_dot=2.*vy+x-(1.-mu)*(x+mu)/(r13**3)-mu*(x-1.+mu)/(r23**3) + fx/m
    vy_dot=-2.*vx+y-(1.-mu)*y/(r13**3)-mu*y/(r23**3) + fy/m
    vz_dot=-(1.-mu)*z/(r13**3)-mu*z/(r23**3) + fz/m

    m_dot=-f_norm/ueq

    s_dot=np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot, m_dot]) #output of CR3BP - 7 components (3+3+1)

    return s_dot




#This function chooses randomly a set of initial conditions (r0,v0,T) for a L1 Halo orbit   
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
    