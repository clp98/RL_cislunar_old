import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from numbalsoda import lsoda_sig, lsoda
from numba import njit, cfunc
from environment.LOR_RL_cislunar.solve_ivp_lsoda import *

#Constants
solar_day = 86400 #solar day, s
mu = 0.01215059 #mass ratio, nondim
l_star = 3.844e+5 #system characteristic lenght, km
t_star = 3.751903e+5 #system characteristic time, s
v_star = l_star/t_star #system characteristic velocity, km/s
g0 = 9.80665e-3 #sea-level gravitational acceleration, km/s^2


def CR3BP_eqs(t, s, f, t_1, t_2, ueq): #with control
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
    x = s[0]
    y = s[1]
    z = s[2]

    #spacecraft velocity
    vx = s[3]
    vy = s[4]
    vz = s[5]

    #spacecraft mass
    m = s[6]

    #Control
    if t >= t_1 and t <= t_2:
        fx = f[0]
        fy = f[1]
        fz = f[2]
        f_norm = norm(f)
    else:
        fx = 0.
        fy = 0.
        fz = 0.
        f_norm = 0.

    #Auxiliary variables
    r13 = sqrt((x+mu)**2 + y**2 + z**2) #Earth-S/C distance
    r23 = sqrt((x-1.+mu)**2 + y**2 + z**2) #Moon-S/C distance

    #Equations of motion
    x_dot = vx
    y_dot = vy
    z_dot = vz

    vx_dot = 2.*vy + x - (1.-mu)*(x+mu)/(r13**3) - mu*(x-1.+mu)/(r23**3) + fx/m
    vy_dot = -2.*vx + y - (1.-mu)*y/(r13**3) - mu*y/(r23**3) + fy/m
    vz_dot = -(1.-mu)*z/(r13**3) - mu*z/(r23**3) + fz/m

    m_dot = - f_norm/ueq

    s_dot = np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot, m_dot]) #output of CR3BP_eqs - 7 components (3+3+1)

    return s_dot


def CR3BP_eqs_free(t, s):  #without control
    """
    Right-hand side of the system of equations of motion of a spacecraft
    in the Earth-Moon circular restricted three body problem (CR3BP).
    The (nondimensioanl) state components are propagated with respect to the system barycenter in the
    Earth-Moon rotating reference frame.

    Args:
        t - (float) time (nondim)
        s - (np.array) spacecraft state (position, velocity, mass) (nondim)

    Return:
        s_dot - (np.array) derivatives of the spacecraft state (nondim)
    """

    #State variables

    #spacecraft position
    x = s[0]
    y = s[1]
    z = s[2]

    #spacecraft velocity
    vx = s[3]
    vy = s[4]
    vz = s[5]

    #Auxiliary variables
    r13 = sqrt((x+mu)**2 + y**2 + z**2) #Earth-S/C distance
    r23 = sqrt((x-1.+mu)**2 + y**2 + z**2) #Moon-S/C distance

    #Equations of motion
    x_dot = vx
    y_dot = vy
    z_dot = vz
    vx_dot = 2.*vy + x - (1.-mu)*(x+mu)/(r13**3) - mu*(x-1.+mu)/(r23**3)
    vy_dot = -2.*vx + y - (1.-mu)*y/(r13**3) - mu*y/(r23**3)
    vz_dot = -(1.-mu)*z/(r13**3) - mu*z/(r23**3)

    s_dot = np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot]) #output of CR3BP_eqs_free - 6 components (3+3)

    return s_dot


@cfunc(lsoda_sig) #A context that controls the transformation of stimulus functions (not strictly necessary)

def CR3BP_eqs_free_lsoda(t, s, sdot, data): #with control but solved with lsoda resolver
    """
    Right-hand side of the system of equations of motion of a spacecraft
    in the Earth-Moon circular restricted three body problem (CR3BP).
    The (nondimensioanl) state components are propagated with respect to the system barycenter in the
    Earth-Moon rotating reference frame.

    Args:
        t - (float) time (nondim)
        s - (np.array) spacecraft state (position, velocity, mass) (nondim)

    Return:
        s_dot - (np.array) derivatives of the spacecraft state (nondim)
    """

    #State variables

    #spacecraft position
    x = s[0]
    y = s[1]
    z = s[2]

    #spacecraft velocity
    vx = s[3]
    vy = s[4]
    vz = s[5]

    #Auxiliary variables
    r13 = sqrt((x+mu)**2 + y**2 + z**2) #Earth-S/C distance
    r23 = sqrt((x-1.+mu)**2 + y**2 + z**2) #Moon-S/C distance

    #Equations of motion
    sdot[0] = vx
    sdot[1] = vy
    sdot[2] = vz
    sdot[3] = 2.*vy + x - (1.-mu)*(x+mu)/(r13**3) - mu*(x-1.+mu)/(r23**3)
    sdot[4] = -2.*vx + y - (1.-mu)*y/(r13**3) - mu*y/(r23**3)
    sdot[5] = -(1.-mu)*z/(r13**3) - mu*z/(r23**3)


def hitMoon(t, y, f, t_1, t_2, ueq):
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

    min_r = 0.005

    rMoon = np.array([1 - mu, 0., 0.])

    r = np.array([y[0], y[1], y[2]])

    dist = norm(r - rMoon) - min_r

    return dist


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


def propagate_cr3bp_free(s, t_eval):
    """
    Propagate CR3BP equation with thrust term

    Args:
        s (np.array): state
        t_eval (list): array of times at which to store the computed solution

    Return:
        sol: new states

    """

    #Integration
    if t_eval[-1] == t_eval[0]:
        return s

    funptr = CR3BP_eqs_free_lsoda.address
    sol, _ = solve_ivp_lsoda(funptr, s, t_eval, 1.0e-07, 1.0e-07)

    return sol 
