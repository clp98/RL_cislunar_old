#Circular Restricted 3 Body Problem

#functions: -CR3BP_equations_controlled
#           -CR3BP_equations_free
#           -Jacobi_constant


import numpy as np
from numpy import sqrt
from numpy.linalg import norm


#important constants
mu=0.012141736 #mass ratio (m2/(m2+m1)), []
l_star=385000 #system chracteristic length, [km]
t_star=3.751903e+5 #system characteristic time, [s]
v_star=l_star/t_star #system characteristic velocity, [km/s]


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


    #control (constant)
    Fx=data[0]
    Fy=data[1]
    Fz=data[2]
    F_mod=norm(np.array([Fx,Fy,Fz]))

    Isp=data[3]
    
    
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
    r13=sqrt((x+mu)**2+y**2+z**2) #earth-sc distance
    r23=sqrt((x-(1-mu))**2+y**2+z**2) #moon-sc distance

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
