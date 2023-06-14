#Bi-Elliptic Restricted 4 Body Problem dynamics

import numpy as np
from numpy import sqrt, cos, sin
from random import random
from numpy.linalg import norm
from environment.frames import *
from environment.pyKepler import *


#Math
conv = np.pi / 180.		                    
g0 = 9.80665e-3  #sea-level gravitational acceleration, km/s^2
SRP_1AU = 4.55987e-06  #solar radiation pressure at 1AU, Pa

#Astrodynamics
solar_day = 86400.  #one solar day [s]
AU = 1.495978707e8  #astronomical unit [km]        
GM_1_d = 132712440018.  #Gravitational parameter  (Sun) [km^3/s^2]
GM_2_d = 398600.  #Gravitational parameter (body 2) [km^3/s^2]
mu = 0.01215059  #Mass ratio for the Halo orbit propagation (the same of equations.py)

#Derive GM_3_d from the value of mu
GM_3_d = (mu * GM_2_d) / (1 - mu)  #Gravitational parameter (body 3) [km^3/s^2]
mu_ref_d = GM_2_d + GM_3_d  #Gravitational parameter (adjoint) [km^3/s^2]
sigma_d = mu_ref_d + GM_1_d

#Orbital elements
a_2_d = AU  #Semi-major axis (body 2-3) [km]
e_2 = 0.017  #Eccentricity (body 2-3)
i_2 = 0  #Inclination of body 2-3's orbit (wrt to the ecliptic) [rad]
p_2_d = a_2_d * (1 - e_2**2)  # Semilatus rectum [km]
omega_2 = 0  #Argument of periapsis [rad]
OMEGA_2 = 0  #RAAN  [rad]
n_2_d = np.sqrt(sigma_d / a_2_d**3)  #Mean motion
a_3_d = 3.844e+5  #Semi-major axis (body 2-3) [km]
e_3 = 0.  #Eccentricity (body 2-3)
i_3 = 23.5*conv  #inclination of body 3 orbit (wrt to ICRS) [rad]
omega_3 = 0.  #Argument of periapsis [rad]
OMEGA_3 = 0.  #RAAN [rad]

#Characteristic values
l_star = a_3_d  #characteristic length [km]
t_star = np.sqrt(a_3_d**3 / mu_ref_d)  #characteristic time [s]
v_star = l_star / t_star  #characteristic velocity [km/s]
a_star = v_star / t_star  #characteristic acceleration [km/s^2]
m_star = 1000.  #characteristic mass [kg]
f_star = m_star * a_star  #characteristic force [kN]

mu_third = GM_3_d 
mu_sys = GM_2_d + GM_3_d
sigma_first = (GM_1_d / mu_sys) + 1.

#Non-dimensional quantities
a_2 = a_2_d / l_star
a_3 = a_3_d / l_star
sigma = sigma_d / mu_ref_d  #Total mass ratio

#Initial orbital parameters of B around body 1
coe_sun = [a_2, e_2, i_2, OMEGA_2, omega_2]

#Initial orbital parameters of body 3 around body 2
coe_moon = [a_3, e_3, i_3, OMEGA_3, omega_3]

A_sc = 1.



def BER4BP_3dof_free(t, X0, data):
    """ 
    ODEs associated to the Bi-Elliptic Retricted 4-Body Problem + Solar Radiation Pressure (SRP)
	The quantities are non-dimensional and expressed in a Radial-Transverse-Normal (RTN) frame, centered in the
    barycenter B of the second and third body.

    Args:
        t (float): time
        X0 (np.array): initial conditions
        data (list):
            R0_ICRS (np.array): initial position vector of B wrt the primary body (3)
            V0_ICRS (np.array): initial velocity vector of B wrt the primary body (3)
            coe3 (list): coes of the third body wrt the second body (excluding the true anomaly) (6)
            with_srp (bool): is Solar radiation pressure considered? (1)
    
    Return:
        X_dot (np.array): derivatives of the state variables
    """

    #Initial conditions
    r_RTN_x = X0[0]
    r_RTN_y = X0[1]
    r_RTN_z = X0[2]
    v_RTN_x = X0[3]
    v_RTN_y = X0[4]
    v_RTN_z = X0[5]
    m = X0[6]
    anu_3 = X0[7]
    
    r_RTN = np.array([r_RTN_x, r_RTN_y, r_RTN_z], dtype=np.float64)

    #Data
    R0_ICRS = data[0:3]
    V0_ICRS = data[3:6]
    coe_3 = data[6:11]

    #Propagation of Keplerian motion of B around P1
    Rt_ICRS, Vt_ICRS = propagate_lagrangian(R0_ICRS, V0_ICRS, t, sigma)
    Rt_RTN = ICRS2RTN(Rt_ICRS, list(coe_3) + [anu_3])

	#Other
    a = coe_3[0]
    e = coe_3[1]
    p = a * (1. - e**2)
    r = p / (1. + e * cos(anu_3))
    h = sqrt(p)

    r_vect = np.array([1., 0., 0.], dtype=np.float64)
    
    anu_3_dot = h / (r**2)
    eta1 = e * sin(anu_3) / (1. + e * cos(anu_3))
    eta2 = e * cos(anu_3) / (1. + e * cos(anu_3))
    eta3 = 1. - eta2
    
    vector_x = -2. * anu_3_dot * (eta1 * v_RTN_x - v_RTN_y) + eta3 * pow(anu_3_dot, 2) * r_RTN_x
    vector_y = -2. * anu_3_dot * (eta1 * v_RTN_y + v_RTN_x) + eta3 * pow(anu_3_dot, 2) * r_RTN_y
    vector_z = -2. * anu_3_dot * eta1 * v_RTN_z - eta2 * pow(anu_3_dot, 2) * r_RTN_z
    vector = np.array([vector_x, vector_y, vector_z])  #velocity vector

    #SRP
    #Sun-S/C distance in AU
    r14 = Rt_RTN + r_RTN
    r14_norm_AU = norm(r14)*l_star/AU

    #SRP at S/C distance
    srp = SRP_1AU/(r14_norm_AU**2)

    #SRP force
    f_srp_norm = srp*A_sc*1e-03/f_star
    f_srp_x = f_srp_norm*r14[0]/norm(r14)
    f_srp_y = f_srp_norm*r14[1]/norm(r14)
    f_srp_z = f_srp_norm*r14[2]/norm(r14)
    f_srp = np.array([f_srp_x, f_srp_y, f_srp_z], dtype=np.float64)

    
	#Derivatives
    r_RTN_x_dot = v_RTN_x
    r_RTN_y_dot = v_RTN_y
    r_RTN_z_dot = v_RTN_z
    v_RTN_dot = vector + (sigma - 1.) / pow(r, 3) * ((1. - mu) * (Rt_RTN - mu * r_vect) / pow(norm(Rt_RTN - mu * r_vect), 3) \
		+ mu * (Rt_RTN + (1. - mu) * r_vect) / pow(norm(Rt_RTN + (1. - mu) * r_vect), 3) \
		- (Rt_RTN + r_RTN) / pow(norm(Rt_RTN + r_RTN), 3)) \
		- (1 - mu) / pow(r, 3) * ((r_RTN + mu * r_vect) / pow(norm(r_RTN + mu * r_vect), 3)) \
		- mu / pow(r, 3) * (r_RTN - (1 - mu) * r_vect) / pow(norm(r_RTN - (1 - mu) * r_vect), 3) + f_srp/m
    m_dot = 0.
        
    X_dot = np.array([r_RTN_x_dot, r_RTN_y_dot, r_RTN_z_dot, \
        v_RTN_dot[0], v_RTN_dot[1], v_RTN_dot[2], \
        m_dot, anu_3_dot], dtype=np.float64)
    
    return X_dot





def BER4BP_3dof(t, X0, data):
    """ 
    ODEs associated to the Bi-Elliptic Retricted 4-Body Problem + Solar Radiation Pressure (SRP)
	The quantities are non-dimensional and expressed in a Radial-Transverse-Normal (RTN) frame, centered in the
    barycenter B of the second and third body.

    Args:
        t (float): time
        X0 (np.array): initial conditions
        data (list):
            f (np.array): thrust vector (3)
            c (float): equivalent ejection velocity (1)
            R0_ICRS (np.array): initial position vector of B wrt the primary body (3)
            V0_ICRS (np.array): initial velocity vector of B wrt the primary body (3)
            coe3 (list): coes of the third body wrt the second body (excluding the true anomaly) (6)
            with_srp (bool): is Solar radiation pressure considered? (1)
    
    Return:
        X_dot (np.array): derivatives of the state variables
    """

    #Initial conditions
    r_RTN_x = X0[0]
    r_RTN_y = X0[1]
    r_RTN_z = X0[2]
    v_RTN_x = X0[3]
    v_RTN_y = X0[4]
    v_RTN_z = X0[5]
    m = X0[6]
    anu_3 = X0[7]
    
    r_RTN = np.array([r_RTN_x, r_RTN_y, r_RTN_z], dtype=np.float64)

    #Data
    f = data[0:3]
    c = data[3]
    R0_ICRS = data[4:7]
    V0_ICRS = data[7:10]
    coe_3 = data[10:15]

    #Propagation of Keplerian motion of B around P1
    Rt_ICRS, Vt_ICRS = propagate_lagrangian(R0_ICRS, V0_ICRS, t, sigma)
    Rt_RTN = ICRS2RTN(Rt_ICRS, list(coe_3) + [anu_3])

	#Other values
    a = coe_3[0]
    e = coe_3[1]
    p = a * (1. - e**2)
    r = p / (1. + e * cos(anu_3))
    h = sqrt(p)

    r_vect = np.array([1., 0., 0.], dtype=np.float64)
    
    anu_3_dot = h / (r**2)
    eta1 = e * sin(anu_3) / (1. + e * cos(anu_3))
    eta2 = e * cos(anu_3) / (1. + e * cos(anu_3))
    eta3 = 1. - eta2
    
    vector_x = -2. * anu_3_dot * (eta1 * v_RTN_x - v_RTN_y) + eta3 * pow(anu_3_dot, 2) * r_RTN_x
    vector_y = -2. * anu_3_dot * (eta1 * v_RTN_y + v_RTN_x) + eta3 * pow(anu_3_dot, 2) * r_RTN_y
    vector_z = -2. * anu_3_dot * eta1 * v_RTN_z - eta2 * pow(anu_3_dot, 2) * r_RTN_z
    vector = np.array([vector_x, vector_y, vector_z])  #velocity vector

    #SRP
    #Sun-S/C distance in AU
    r14 = Rt_RTN + r_RTN
    r14_norm_AU = norm(r14)*l_star/AU

    #SRP at S/C distance
    srp = SRP_1AU/(r14_norm_AU**2)

    #SRP force
    f_srp_norm = srp*A_sc*1e-03/f_star
    f_srp_x = f_srp_norm*r14[0]/norm(r14)
    f_srp_y = f_srp_norm*r14[1]/norm(r14)
    f_srp_z = f_srp_norm*r14[2]/norm(r14)
    f_srp = np.array([f_srp_x, f_srp_y, f_srp_z], dtype=np.float64)

    
	#Derivatives
    r_RTN_x_dot = v_RTN_x
    r_RTN_y_dot = v_RTN_y
    r_RTN_z_dot = v_RTN_z
    v_RTN_dot = vector + (sigma - 1.) / pow(r, 3) * ((1. - mu) * (Rt_RTN - mu * r_vect) / pow(norm(Rt_RTN - mu * r_vect), 3) \
		+ mu * (Rt_RTN + (1. - mu) * r_vect) / pow(norm(Rt_RTN + (1. - mu) * r_vect), 3) \
		- (Rt_RTN + r_RTN) / pow(norm(Rt_RTN + r_RTN), 3)) \
		- (1 - mu) / pow(r, 3) * ((r_RTN + mu * r_vect) / pow(norm(r_RTN + mu * r_vect), 3)) \
		- mu / pow(r, 3) * (r_RTN - (1 - mu) * r_vect) / pow(norm(r_RTN - (1 - mu) * r_vect), 3) + f_srp/m + f/m
    m_dot = - norm(f*r)/c
        
    X_dot = np.array([r_RTN_x_dot, r_RTN_y_dot, r_RTN_z_dot, \
        v_RTN_dot[0], v_RTN_dot[1], v_RTN_dot[2], \
        m_dot, anu_3_dot], dtype=np.float64)
    
    return X_dot

