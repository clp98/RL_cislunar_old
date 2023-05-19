
#Circular Restricted 3 Body Problem

#functions: -CR3BP_equations_controlled
#           -CR3BP_equations_free
#           -Jacobi_constant


import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from environment.BER4BP import *
#from environment.pyKepler import *


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




def CR3BP_equations_controlled_ivp(t, state, data):  #with control
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
    F=data[0:3]
    ueq=data[3]
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



def hitMoon(t, y, data):
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




 #different dist_r approach - events function
def min_dist_r(t, y, data):  #minimum distance in position and velocity from the halo

    r_Halo = y[0:3]
    v_Halo = y[3:6]

    #sc absolute and relative velocities (wrt Halo)
    r_sc = data[0:3]  #absolute sc velocity
    r_rel_Halo = r_Halo - r_sc  #relative sc velocity

    r_rel_Halo_dot = dot(v_Halo, r_rel_Halo)/(norm(r_rel_Halo))
   

    return r_rel_Halo_dot
        


def min_dist_v(t, y, data):

    v_Halo = y[3:6]

    #sc absolute and relative velocities (wrt Halo)
    v_sc = data[3:6] #np.array([state['v'][0], state['v'][1], state['v'][2]])  #absolute sc velocity
    v_rel_Halo = v_Halo - v_sc  #relative sc velocity

    a_Halo = CR3BP_equations_ivp(t, y, data)[3:6]

    v_rel_Halo_dot = dot(a_Halo, v_rel_Halo)/(norm(v_rel_Halo))


    return v_rel_Halo_dot



def CR3BP_equations_ivp(t, state, data):  #with control
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

    #additional quantities
    r13=sqrt((x+mu)**2+y**2+z**2) #earth-sc distance
    r23=sqrt((x-(1-mu))**2+y**2+z**2) #moon-sc distance

    #obtain derivative of the state
    state_dot=np.zeros(6)
    state_dot[0]=vx
    state_dot[1]=vy
    state_dot[2]=vz

    state_dot[3]=2*vy+x-((1-mu)*(x+mu))/(r13**3)-(mu*(x-1+mu))/(r23**3) 
    state_dot[4]=-2*vx+y-((1-mu)*y)/(r13**3)-(mu*y)/(r23**3) 
    state_dot[5]=z*((-(1-mu)/(r13**3))+(-mu/(r23**3))) 

    return state_dot


def Jacobi_const(x, y, z, vx, vy, vz):
    """
    Evaluate the Jacobi constant C.

    Args:
        x, y, z: spacecraft position (nondim)
        vx, vy, vz: spacecraft velocity (nondim)

    Return:
        C: Jacobi constant
    """

    #Auxiliary variables
    r13 = sqrt((x+mu)**2 + y**2 + z**2) #Earth-S/C distance
    r23 = sqrt((x-1.+mu)**2 + y**2 + z**2) #Moon-S/C distance

    #Jacobi constant
    C = 2.*((1. - mu)/r13 + mu/r23 + 0.5*(x**2 + y**2)) - (vx**2 + vy**2 + vz**2)

    return C
