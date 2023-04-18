import numpy as np
from numpy import sqrt, cos, sin
from random import random
from numpy.linalg import norm
from environment.frames import *
from environment.pyKepler import *


# Math
conv = np.pi / 180.		                    # Deg to rad
g0 = 9.80665e-3                             # sea-level gravitational acceleration, km/s^2
# physical
M = 5.4e11
d2 = 0.78
d3 = 0.17
d_ratio = d2/d3
m3 = M/(d_ratio**3+1)
m2 = M-m3
R_2_d = 0.5*d2                                # Radius of second body, km
R_3_d = 0.5*d3                                # Radius of third body, km
SRP_1AU = 4.55987e-06                         # solar radiation pressure at 1AU, Pa                                   # Area of the S/C solar panels, m^2
# Astrodynamics
solar_day = 86400.                          # one solar day [s]
AU = 1.495978707e8                          # astronomical unit [km]
G = 6.67430e-20                             # universal gravitational constant [km^3⋅kg–1⋅s–2]
GM_1_d = 132712440018.	                    # Gravitational parameter  (Sun) [km^3/s^2]
GM_2_d = G*m2			                    # Gravitational parameter (body 2) [km^3/s^2]
GM_3_d = G*m3			                    # Gravitational parameter (body 3) [km^3/s^2]
mu_ref_d = GM_2_d + GM_3_d	                # Gravitational parameter (adjoint) [km^3/s^2]
sigma_d = mu_ref_d + GM_1_d
# Orbital elements
a_2_d = 1.6443941966*AU				                                    # Semi-major axis (body 2-3) [km]
e_2 = 0.3836974806			                                            # Eccentricity (body 2-3)
i_2 = 3.4077057 * conv	#0.		                                        # Inclination of body 2-3's orbit (wrt to the ecliptic) [rad]
p_2_d = a_2_d * (1 - e_2**2)	                                        # Semilatus rectum [km]
omega_2 = 319.318868 * conv			                                    # Argument of periapsis [rad]
OMEGA_2 = 73.199435 * conv			                                    # RAAN  [rad]
n_2_d = np.sqrt(sigma_d / a_2_d**3)                                     # Mean motion
XM_2ws = (136.6502278*conv + n_2_d*(453*solar_day)) % (2.*np.pi)                    # Mean anomaly (25 September 2022, 23:59)
anu_2ws = AE2anu(e_2, XM2AE(e_2, XM_2ws))

a_3_d = 1.19				                # Semi-major axis (body 2-3) [km]
e_3 = 0.			                        # Eccentricity (body 2-3)
i_3 = 160. * conv	          		        # Inclination of body 3 orbit (wrt to ICRS) [rad]
omega_3 = 0. * conv			                # Argument of periapsis [rad]
OMEGA_3 = 149. * conv			            # RAAN [rad]

# Conversion units (3-DoF)
rconv = a_3_d								# Length [km]
tconv = np.sqrt(a_3_d**3 / mu_ref_d)	    # Time [s]
vconv = rconv / tconv						# Velocity [km/s]
aconv = vconv / tconv                       # Acceleration [km/s^2]
mconv = 1000.                               # Mass [kg]
fconv = mconv * aconv                       # Force [kN]



mu_third = GM_3_d  #valori giusti?
mu_sys = GM_2_d + GM_3_d
sigma_first = (GM_1_d / mu_sys) + 1.


anu_3=random.random(0,180)  #moon (third body) true anomaly choosen randomly (va scalato??)

sun_r0_x =
sun_r0_y =
sun_r0_z =
sun_v0_x =
sun_v0_y =
sun_v0_z =
sun_r0 = np.array(sun_r0_x, sun_r0_y, sun_r0_z) 
sun_v0 = np.array(sun_v0_x, sun_v0_y, sun_v0_z)


################################################################################################################


X0=[s[0], s[1], s[2], s[3], s[4], s[5], anu_3, self.m_SC]

data=[F_vect[0], F_vect[1], F_vect[2], 9.80665e-3*self.Isp, sun_r0[0], sun_r0[1], sun_r0[2], /
      sun_v0[0], sun_v0[1], sun_v0[2], mu_third, mu_sys, sigma_first, /
      coe_3[0], coe_3[1], coe_3[2], coe_3[3], coe_3[4], coe_3[5], /
      False]

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

    # Initial conditions
    r_RTN_x = X0[0]
    r_RTN_y = X0[1]
    r_RTN_z = X0[2]
    v_RTN_x = X0[3]
    v_RTN_y = X0[4]
    v_RTN_z = X0[5]
    m = X0[6]
    anu_3 = X0[7]
    
    r_RTN = np.array([r_RTN_x, r_RTN_y, r_RTN_z], dtype=np.float64)

    # Data
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
    Rt_ICRS, Vt_ICRS = propagate_lagrangian(R0_ICRS, V0_ICRS, t, sigma_first)
    Rt_RTN = ICRS2RTN(Rt_ICRS, list(coe_3) + [anu_3])

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
    vector = np.array([vector_x, vector_y, vector_z])  #velocity vector

    #SRP
    if with_srp:

        #Sun-S/C distance in AU
        r14 = Rt_RTN + r_RTN
        r14_norm_AU = norm(r14)*rconv/AU

        #SRP at S/C distance
        srp = SRP_1AU/(r14_norm_AU**2)

        #SRP force
        f_srp_norm = srp*self.A_sc*1e-03/fconv
        f_srp_x = f_srp_norm*r14[0]/norm(r14)
        f_srp_y = f_srp_norm*r14[1]/norm(r14)
        f_srp_z = f_srp_norm*r14[2]/norm(r14)
        f_srp = np.array([f_srp_x, f_srp_y, f_srp_z], dtype=np.float64)

    else:
        f_srp = np.zeros(3, dtype=np.float64)

    
	#Derivatives
    r_RTN_x_dot = v_RTN_x
    r_RTN_y_dot = v_RTN_y
    r_RTN_z_dot = v_RTN_z
    v_RTN_dot = vector + (sigma_first - 1.) / pow(r, 3) * ((1. - mu_third) * (Rt_RTN - mu_third * r_vect) / pow(norm(Rt_RTN - mu_third * r_vect), 3) \
		+ mu_third * (Rt_RTN + (1. - mu_third) * r_vect) / pow(norm(Rt_RTN + (1. - mu_third) * r_vect), 3) \
		- (Rt_RTN + r_RTN) / pow(norm(Rt_RTN + r_RTN), 3)) \
		- (1 - mu_third) / pow(r, 3) * ((r_RTN + mu_third * r_vect) / pow(norm(r_RTN + mu_third * r_vect), 3)) \
		- mu_third / pow(r, 3) * (r_RTN - (1 - mu_third) * r_vect) / pow(norm(r_RTN - (1 - mu_third) * r_vect), 3) + f_srp/m + f/m
    m_dot = - norm(f*r)/c
        
    X_dot = np.array([r_RTN_x_dot, r_RTN_y_dot, r_RTN_z_dot, \
        v_RTN_dot[0], v_RTN_dot[1], v_RTN_dot[2], \
        anu_3_dot, m_dot], dtype=np.float64)
    
    return X_dot
