
#Circular Restricted 3 Body Problem

#functions: -CR3BP_equations_controlled
#           -CR3BP_equations_free
#           -Jacobi_constant


import numpy as np
from numpy import sqrt
from numpy.linalg import norm
#from environment.pyKepler import *


#important constants
solar_day = 86400  #solar day, s
mu = 0.01215059  #mass ratio, nondim
l_star = 3.844e+5  #system characteristic lenght, km
t_star = 3.751903e+5  #system characteristic time, s
v_star = l_star/t_star  #system characteristic velocity, km/s
a_star = v_star/t_star  #system characteristic acceleration, km\s^2
m_star = 1000.  #system characteristic mass, kg
f_star = m_star*a_star  
g0 = 9.80665e-3  #sea-level gravitational acceleration, km/s^2


def CR3BP_equations_controlled(t, state, state_dot, data):  #with control
    '''
    input: -t (time)
           -state
           -F (thrust)
           -t_F_i (start of thrusting period) 
           -t_F_f (end of thrusting period)
           -Isp (specific impulse)
    
    output: -state_dot (derivative of the state)

    '''

    #state variables (state=[x y z vx vy vz m])

    #spacecraft position
    x=state[0]
    y=state[1]
    z=state[2]

    #spacecraft velocity
    vx=state[3]
    vy=state[4]
    vz=state[5]

    #spacecraft mass
    m=state[6]


    #control (assumed to be constant)
    Fx=data[0][0]
    Fy=data[0][1]
    Fz=data[0][2]
    F_mod=norm(np.array([Fx,Fy,Fz]))

    Isp=data[1]
    
    
    #additional quantities
    r13=sqrt((x+mu)**2+y**2+z**2) #earth-sc distance
    r23=sqrt((x-(1-mu))**2+y**2+z**2) #moon-sc distance

    #obtain derivative of the state
    state_dot[0]=vx
    state_dot[1]=vy
    state_dot[2]=vz

    state_dot[3]=2*vy+x-((1-mu)*(x+mu))/(r13**3)-(mu*(x-1+mu))/(r23**3) + Fx/m
    state_dot[4]=-2*vx+y-((1-mu)*y)/(r13**3)-(mu*y)/(r23**3) + Fy/m
    state_dot[5]=z*((-(1-mu)/(r13**3))+(-mu/(r23**3))) + Fz/m

    state_dot[6]=-F_mod/Isp




def CR3BP_equations_controlled_ivp(t, state, F, ueq):  #with control
    '''
    input: -t (time)
           -state
           -F (thrust)
           -t_F_i (start of thrusting period) 
           -t_F_f (end of thrusting period)
           -Isp (specific impulse)
    
    output: -state_dot (derivative of the state)

    '''

    #state variables (state=[x y z vx vy vz m])

    #spacecraft position
    x=state[0]
    y=state[1]
    z=state[2]

    #spacecraft velocity
    vx=state[3]
    vy=state[4]
    vz=state[5]

    #spacecraft mass
    m=state[6]


    #control (assumed to be constant)
    Fx=F[0]
    Fy=F[1]
    Fz=F[2]
    F_mod=norm(F)
    
    
    #additional quantities
    r13=sqrt((x+mu)**2+y**2+z**2) #earth-sc distance
    r23=sqrt((x-(1-mu))**2+y**2+z**2) #moon-sc distance

    #obtain derivative of the state
    state_dot=np.zeros(7)
    state_dot[0]=vx
    state_dot[1]=vy
    state_dot[2]=vz

    state_dot[3]=2*vy+x-((1-mu)*(x+mu))/(r13**3)-(mu*(x-1+mu))/(r23**3) + Fx/m #+ f_srp_x/m
    state_dot[4]=-2*vx+y-((1-mu)*y)/(r13**3)-(mu*y)/(r23**3) + Fy/m #+ f_srp_y/m
    state_dot[5]=z*((-(1-mu)/(r13**3))+(-mu/(r23**3))) + Fz/m #+ f_srp_z/m

    state_dot[6]=-F_mod/ueq


    '''#SRP disturbance
    if self.with_srp:
    
        mu_sun = 132712440018.
        r0_sun = np.random.uniform(-100000,100000,3)  #choose randomly initial position vector
        v0_sun = np.random.uniform(-100000,100000,3)  #choose randomly initial velocity vector

        r_sun, v_sun = propagate_lagrangian(r0_sun, v0_sun, t, mu_sun)  #propagate sun orbit
        r = np.array([x, y, z])
        r14 = r-r_sun  #obtain the sun-sc position vector
    
        r14_AU = r14/1.495978707e8  #r14 in AU
        SRP_1AU = 4.55987e-06  #solar radiation pressure at 1 AU [Pa]

        #SRP at sc distance
        srp = SRP_1AU/norm(r14_AU)**2

        #SRP force
        f_srp_norm = srp*self.A_sc*1e-03/f_star
        f_srp_x = f_srp_norm*r14[0]/norm(r14)
        f_srp_y = f_srp_norm*r14[1]/norm(r14)
        f_srp_z = f_srp_norm*r14[2]/norm(r14)
        f_srp = np.array([f_srp_x, f_srp_y, f_srp_z], dtype=np.float64)

    else:
        f_srp = np.zeros(3, dtype=np.float64)'''




    return state_dot




def CR3BP_equations_free(t, state, state_dot, data):  #without control
    '''
    input: -t (time)
           -state
    
    output: -state_dot (derivative of the state)

    '''

    #state variables (state=[x y z vx vy vz])

    #spacecraft position
    x=state[0]
    y=state[1]
    z=state[2]

    #spacecraft velocity
    vx=state[3]
    vy=state[4]
    vz=state[5]

    #additional quantities
    r13=sqrt((x+mu)**2+y**2+z**2)  #earth-sc distance
    r23=sqrt((x-(1-mu))**2+y**2+z**2)  #moon-sc distance

    #obtain derivative of the state
    state_dot[0]=vx
    state_dot[1]=vy
    state_dot[2]=vz

    state_dot[3]=2*vy+x-((1-mu)*(x+mu))/(r13**3)-(mu*(x-1+mu))/(r23**3)
    state_dot[4]=-2*vx+y-((1-mu)*y)/(r13**3)-(mu*y)/(r23**3)
    state_dot[5]=z*((-(1-mu)/(r13**3))+(-mu/(r23**3)))






def Jacobi_constant(x, y, z, vx, vy, vz):
    '''
    input: -x,y,z (spacecraft position coordinates)
           -vx,vy,vz (spacecraft velocuty coordinates)
    output: -Jacobi constant
    
    '''
    
    r13=sqrt((x+mu)**2+y**2+z**2) #earth-sc distance
    r23=sqrt((x-(1-mu))**2+y**2+z**2) #moon-sc distance
    
    C=2*(((1-mu)/r13)+mu/r23+0.5*(x**2+y**2+z**2))-(vx**2+vy**2+vz**2)
    
    return C

def hitMoon(t, y, f, ueq):
    """
    Check if the spacecraft is too close to the Moon

    Args:
        t (float): time
        y (list): solution
        f (list): thrust
        ueq (float): equivalent ejection velocity

    Return:
        dist (float): how far we are from the minimum
            distance from the Moon
    """

    min_r = 0.01

    rMoon = np.array([1 - mu, 0., 0.])

    r = np.array([y[0], y[1], y[2]])

    dist = norm(r - rMoon) - min_r

    return dist
