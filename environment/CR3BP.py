import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from scipy.integrate import solve_ivp

#Constants
solar_day = 86400 #solar day, s
mu = 0.01215059 #mass ratio, nondim
l_star = 3.844e+5 #system characteristic lenght, km
t_star = 3.751903e+5 #system characteristic time, s
v_star = l_star/t_star #system characteristic velocity, km/s
g0 = 9.80665e-3 #sea-level gravitational acceleration, km/s^2


def CR3BP_eqs(t, s, f, t_1, t_2, ueq):
    """
    Right-hand side of the system of equations of motion of a spacecraft
    in the Earth-Moon circular restricted three body problem (CR3BP),
    with low thrust terms.
    The (nondimensioanl) state components are propagated with respect to the system barycenter in the
    Earth-Moon rotating reference frame.

    Args:
        t - (float) time (nondim)
        s - (np.array) spacecraft state (position, velocity, mass) (nondim)
        f - (np.array) spacecraft thrust (nondim)
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

    s_dot = np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot, m_dot], dtype='object')

    return s_dot


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


def propagate_cr3bp_free(r, v, t_span, t_eval):
    """
    Propagate CR3BP equation with thrust term

    Args:
        r (np.array): position
        v (np.array): velocity
        t_span (list): temporal span ([t0, tf])
        t_eval (list): array of times at which to store the computed solution, 
            must lie within `t_span`
    Return:
        r_new: new positions
        v_new: new velocities

    """
    
    #State
    s = np.array([r[0], r[1], r[2], \
        v[0], v[1], v[2], 1.], dtype=np.float32)

    #Integration
    sol = solve_ivp(fun=CR3BP_eqs, t_span=t_span, y0=s, method='RK45', t_eval=t_eval, \
            args=(np.array([0., 0., 0.]), t_span[0], t_span[1], 1.), rtol=1e-8, atol=1e-8)

    #Next state
    r_new = np.transpose(np.array(sol.y[0:3][:], dtype=np.float32))
    v_new = np.transpose(np.array(sol.y[3:6][:], dtype=np.float32))

    return r_new, v_new
    