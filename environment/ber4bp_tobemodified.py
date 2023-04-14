import numpy as np
from scipy.integrate import solve_ivp
from environment.rk4 import *

def BER4BP_3dof(t, X0, data):
    """ 
    ODEs associated to the Bi-Elliptic Retricted 4-Body Problem + Solar Radiation Pressure (SRP)
	The quantities are non-dimensional and expressed in a Radial-Transverse-Normal (RTN) frame, centered in the
    barycenter B of the second and third body.

    Args:
        t (float): time
        X0 (np.array): initial conditions
        data (list):
            f (np.array): thrust vector
            c (float): equivalent ejection velocity
            R0_ICRS (np.array): initial position vector of B wrt the primary body
            V0_ICRS (np.array): initial velocity vector of B wrt the primary body
            coe3 (list): coes of the third body wrt the second body (excluding the true anomaly)
            with_srp (bool): is Solar radiation pressure considered?
    
    Return:
        X_dot (np.array): derivatives of the state variables
    """

    #Initial conditions
    r_x = X0[0]
    r_y = X0[1]
    r_z = X0[2]
    v_x = X0[3]
    v_y = X0[4]
    v_z = X0[5]
    anu_3 = X0[6]
    m = X0[7]

    r= np.array([r_x, r_y, r_z], dtype=np.float64)

    #Data
    f = data[0:3]
    c = data[3]
    R0_ICRS = data[4:7]
    V0_ICRS = data[7:10]
    mu_third = data[10]
    mu_sys = data[11]
    sigma_first = data[12]
    coe_3 = data[13:18]
    with_srp = data[18]

    # Propagation of Keplerian motion of B around P1
    #Rt_ICRS, _ = propagate_lagrangian(R0_ICRS, V0_ICRS, t, sigma_first)
    #Rt_RTN = ICRS2RTN(Rt_ICRS, list(coe_3) + [anu_3])

	# Other
    a = coe_3[0]
    e = coe_3[1]
    p = a * (1. - e**2)
    r = p / (1. + e * cos(anu_3))
    h = sqrt(p*mu_sys)

    r_vect = np.array([1., 0., 0.], dtype=np.float64)
    
    anu_3_dot = h / (r**2)
    eta1 = e * sin(anu_3) / (1. + e * cos(anu_3))
    eta2 = e * cos(anu_3) / (1. + e * cos(anu_3))
    eta3 = 1. - eta2
    
    vector_x = -2. * anu_3_dot * (eta1 * v_RTN_x - v_RTN_y) + eta3 * pow(anu_3_dot, 2) * r_RTN_x
    vector_y = -2. * anu_3_dot * (eta1 * v_RTN_y + v_RTN_x) + eta3 * pow(anu_3_dot, 2) * r_RTN_y
    vector_z = -2. * anu_3_dot * eta1 * v_RTN_z - eta2 * pow(anu_3_dot, 2) * r_RTN_z
    vector = np.array([vector_x, vector_y, vector_z])



    #Solar radiation pressure perturbation
    if with_srp:

        # Sun-S/C distance in AU
        r14 = 
        r14_norm_AU = norm(r14)*rconv/AU

        # SRP at S/C distance
        srp = SRP_1AU/(r14_norm_AU**2)

        # SRP force
        f_srp_norm = srp*A_SC*1e-03/fconv
        f_srp_x = f_srp_norm*r14[0]/norm(r14)
        f_srp_y = f_srp_norm*r14[1]/norm(r14)
        f_srp_z = f_srp_norm*r14[2]/norm(r14)
        f_srp = np.array([f_srp_x, f_srp_y, f_srp_z], dtype=np.float64)
    else:
        f_srp = np.zeros(3, dtype=np.float64)

    
	# Derivatives
    r_x_dot = v_x
    r_y_dot = v_y
    r_z_dot = v_z
    v_dot = vector + (sigma_first - 1.) / pow(r, 3) * ((1. - mu_third) * (Rt_RTN - mu_third * r_vect) / pow(norm(Rt_RTN - mu_third * r_vect), 3) \
		+ mu_third * (Rt_RTN + (1. - mu_third) * r_vect) / pow(norm(Rt_RTN + (1. - mu_third) * r_vect), 3) \
		- (Rt_RTN + r_RTN) / pow(norm(Rt_RTN + r_RTN), 3)) \
		- (1 - mu_third) / pow(r, 3) * ((r_RTN + mu_third * r_vect) / pow(norm(r_RTN + mu_third * r_vect), 3)) \
		- mu_third / pow(r, 3) * (r_RTN - (1 - mu_third) * r_vect) / pow(norm(r_RTN - (1 - mu_third) * r_vect), 3) + f_srp/m + f/m
    m_dot = - norm(f*r)/c
        
    X_dot = np.array([r_RTN_x_dot, r_RTN_y_dot, r_RTN_z_dot, \
        v_RTN_dot[0], v_RTN_dot[1], v_RTN_dot[2], \
        anu_3_dot, m_dot], dtype=np.float64)
    
    return X_dot







##########################################################################
Y = []

def bielliptic_four_body(t, Y, mu):

    # Unpack the state vector
    x1, x2, x3, x4, y1, y2, y3, y4 = Y
    
    #distances between the bodies
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r14 = np.sqrt((x4 - x1)**2 + (y4 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    r24 = np.sqrt((x4 - x2)**2 + (y4 - y2)**2)
    r34 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
    
    #calculate the accelerations of each body
    x1_dotdot = mu * ((x2 - x1) / r12**3 + (x3 - x1) / r13**3 + (x4 - x1) / r14**3)
    x2_dotdot = mu * ((x1 - x2) / r12**3 + (x3 - x2) / r23**3 + (x4 - x2) / r24**3)
    x3_dotdot = mu * ((x1 - x3) / r13**3 + (x2 - x3) / r23**3 + (x4 - x3) / r34**3)
    x4_dotdot = mu * ((x1 - x4) / r14**3 + (x2 - x4) / r24**3 + (x3 - x4) / r34**3)
    y1_dotdot = mu * ((y2 - y1) / r12**3 + (y3 - y1) / r13**3 + (y4 - y1) / r14**3)
    y2_dotdot = mu * ((y1 - y2) / r12**3 + (y3 - y2) / r23**3 + (y4 - y2) / r24**3)
    y3_dotdot = mu * ((y1 - y3) / r13**3 + (y2 - y3) / r23**3 + (y4 - y3) / r34**3)
    y4_dotdot = mu * ((y1 - y4) / r14**3 + (y2 - y4) / r24**3 + (y3 - y4) / r34**3)
    
    Y_dot = np.array([x1_dotdot, x2_dotdot, x3_dotdot, x4_dotdot, y1_dotdot, y2_dotdot, y3_dotdot, y4_dotdot])
    
    return Y_dot

    #integration

  














