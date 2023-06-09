#Functions for the 4 body dynamics, taken from DART paper
import numpy as np
from numpy import sqrt, sin, cos, tan, sinh, cosh, tanh, abs, \
    arcsin, arccos, arctan, arctanh, arctan2, dot, cross
from numpy.linalg import norm
from numba import njit



def sph2cart(r, theta, phi):
    
    x = r*cos(theta)*cos(phi)
    y = r*sin(theta)*cos(phi)
    z = r*sin(phi)

    return x, y, z


def cart2sph(x, y, z):

    r = sqrt(x**2 + y**2 + z**2)
    theta = arctan2(y, x)
    phi = arctan(z/sqrt(x**2 + y**2))

    return r, theta, phi


""" RTN to ECI """
def RTNtoECI(r, vec_rtn):
    """
    :return vec_eci: vec in eci frame

    """
    
    vers_r = r/norm(r)
    vec_t = cross(np.array([0., 0., 1.], dtype=np.float64), vers_r)
    vers_t = vec_t/norm(vec_t) 
    vers_n = cross(vers_r, vers_t)
    
    vec_eci = vec_rtn[0]*vers_r + \
        vec_rtn[1]*vers_t + \
        vec_rtn[2]*vers_n

    return vec_eci


""" RTN to ECI """
def ECItoRTN(r, vec_eci):
    """
    :return vec_eci: vec in eci frame

    """
    
    vers_r = r/norm(r)
    vec_t = cross(np.array([0., 0., 1.], dtype=np.float64), vers_r)
    vers_t = vec_t/norm(vec_t) 
    vers_n = cross(vers_r, vers_t) 

    vec_rtn = np.array([dot(vec_eci, vers_r), dot(vec_eci, vers_t), dot(vec_eci, vers_n)], dtype=np.float64)

    return vec_rtn


def polar_rfvf(rT0, rf, vf, tf):
    _, thetaT0, _ = cart2sph(rT0[0], rT0[1], rT0[2])
    _, _, phif = cart2sph(rf[0], rf[1], rf[2])
    thetaf = thetaT0 + (norm(vf)/norm(rf))*tf
    rf_polar = np.array([norm(rf), thetaf, phif], dtype=np.float64)
    vf_polar = ECItoRTN(rf, vf)
    
    return rf_polar, vf_polar
    

def polar_rv(r, v):
    _, theta, phi = cart2sph(r[0], r[1], r[2])
    v_pol = ECItoRTN(r, v)
    r_pol = np.array([norm(r), theta, phi], dtype=np.float64)

    return r_pol, v_pol


def cartesian_rv(r, v):
    x, y, z = sph2cart(r[0], r[1], r[2])
    r_cart = np.array([x, y, z], dtype=np.float64)
    v_cart = RTNtoECI(r_cart, v)

    return r_cart, v_cart


def AE2anu(e, AE):
    # input:
    #   AE : eccentric anomaly, rad  
    #
    # output:
    #   anu : true anomaly, rad
    
    if (e < 1):
        # ----------------------Elliptical----------------------
        anu = 2 * arctan2(sqrt(1 + e)*sin(0.5*AE), sqrt(1 - e)*cos(0.5*AE))
        if (anu < 0):
            anu += 2 * np.pi
    else:
        # ----------------------Hyperbolic----------------------
        anu = 2. * arctan(sqrt((e + 1) / (e - 1))*tanh(AE / 2))
    
    return anu


def anu2AE(e, anu):
    # input:
    #   anu : true anomaly, rad
    # output:
    #   AE : eccentric anomaly, rad 
    
    if (e < 1):
        AE = 2 * arctan(sqrt((1 - e) / (1 + e))*tan(anu / 2))
        if (AE < 0):
            AE += 2 * np.pi
    else:
        AE = 2 * arctanh(sqrt((e - 1) / (e + 1))*tan(anu / 2))
    
    return AE


def AE2XM(e, AE):
    # input:
    #   AE : eccentric anomaly, rad
    # output:
    #   XM : mean anomaly, rad

    if (e < 1):
        XM = AE - e*sin(AE)
        if (XM < 0):
            XM += 2 * np.pi
    else:
        XM = e*sinh(AE) - AE
    
    return XM


def XM2AE(e, XM):
    # input:
    #   XM : mean anomaly, rad
    # output:
    #   AE : eccentric anomaly, rad
    
    err = 1.
    eps = 1.0e-10
    iter = 1
    max_iter = 1000
    
    AE0 = XM
    AE = 0.
    if (e < 1):
        while (iter < max_iter and err > eps):
            AE = AE0 - (AE0 - e * sin(AE0) - XM) / (1 - e * cos(AE0))
            err = abs(AE - AE0)
            AE0 = AE
            iter += 1
            if (AE < 0):
                AE += 2 * np.pi
    else:
        while (iter < max_iter and err > eps):
            AE = AE0 - (-AE0 + e * sinh(AE0) - XM) / (-1 + e * cosh(AE0))
            err = abs(AE - AE0)
            AE0 = AE
            iter += 1
    
    return AE


def anu_propagator(anu0, dt, e, a, amu):
    """
    True anomaly propagation for a time dt from anu0 to anu along an orbit of eccentricity e and semi-major axis a
    """
    AE0 = anu2AE(e, anu0)
    XM0 = AE2XM(e, AE0)
    n = sqrt(amu/pow(abs(a),3))
    t0 = XM0 / n
    t1 = t0 + dt
    XM1 = n * t1
    AE1 = XM2AE(e, XM1)

    return AE2anu(e, AE1)

@njit
def stumpS(z):
    """
    This function evaluates the Stumpff function S(z) according
    to Equation 3.49.

    z - input argument
    s - value of S(z)

    User M - functions required : none
      ------------------------------------------------------------*/
    """
    if z > 0:
        s = (sqrt(z) - sin(sqrt(z))) / (sqrt(z)**3)
    elif z < 0:
        s = (sinh(sqrt(-z)) - sqrt(-z)) / (sqrt(-z) ** 3)
    else:
        s = 1. / 6.
    return s

@njit
def stumpC( z):
    """
    This function evaluates the Stumpff function C(z) according
	to Equation 3.50.
		z - input argument
	    c - value of C(z)
	User M - functions required : none
	------------------------------------------------------------
    :param z:
    :return: double
    """
    if (z > 0):
        c = (1 - cos(sqrt(z))) / z
    elif (z < 0):
        c = (cosh(sqrt(-z)) - 1) / (-z)
    else:
        c = 0.5
    return c

@njit
def kepler_U(mu,  dt,  ro,  vro,  ua):
    """
    This function uses Newton's method to solve the universal
    Kepler equation for the universal anomaly.

    mu - gravitational parameter(kmˆ3 / sˆ2)
    x - the universal anomaly(kmˆ0.5)
    dt - time since x = 0 (s)
    ro - radial position(km) when x = 0
    vro - radial velocity(km / s) when x = 0
    ua - reciprocal of the semimajor axis(1 / km)
    z - auxiliary variable(z = a*xˆ2)
    C - value of Stumpff function C(z)
    S - value of Stumpff function S(z)
    n - number of iterations for convergence
    nMax - maximum allowable number of iterations

    User M - functions required : stumpC, stumpS
    ------------------------------------------------------------
    """

    # //%...Set an error tolerance and a limit on the number of
	# //% iterations :
    ERROR = 1e-8
    N_MAX = 10000

    # //%...Starting value for x:
    x = sqrt(mu)*abs(ua)*dt
    # //%...Iterate on Equation 3.62 until convergence occurs within the error tolerance :
    n = 0
    ratio = 1

    while abs(ratio) > ERROR and n <= N_MAX:
        n = n + 1
        C = stumpC(ua*x*x)
        S = stumpS(ua*x*x)
        F = ro*vro / sqrt(mu)*x*x*C + (1. - ua*ro)*pow(x,3)*S + ro*x - sqrt(mu)*dt
        dFdx = ro*vro / sqrt(mu)*x*(1. - ua*x*x*S) + (1. - ua*ro)*x*x*C + ro
        ratio = F / dFdx
        x = x - ratio

    if (n > N_MAX):
        print("Number of iterations of Kepler''s equation = " , n , " > " , N_MAX)

    return x

@njit
def lagrangefg_kepU(mu, x, t, ro, ua):
    """"
    ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
    function[f, g] = f_and_g(x, t, r0, a)
    ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜

    This function calculates the Lagrange f and g coefficients.

    mu - the gravitational parameter(kmˆ3 / sˆ2)
    ua - reciprocal of the semimajor axis(1 / km)
    r0 - the radial position at time t(km)
    t - the time elapsed since t(s)
    x - the universal anomaly after time t(kmˆ0.5)
    f - the Lagrange f coefficient(dimensionless)
    g - the Lagrange g coefficient(s)

    User M - functions required : stumpC, stumpS
    ------------------------------------------------------------
    """

    x2 = x * x
    z = ua * x2
    # // %...Equation 3.66a e  3.66b:
    f = 1 - x2 / ro * stumpC(z)
    g = t - 1. / sqrt(mu) * pow(x, 3) * stumpS(z)
    return f, g

@njit
def lagrangefDotgDot_kepU(mu, x, r, ro, ua):
    """
    ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
    function[fdot, gdot] = fDot_and_gDot(x, r, ro, a)
    ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜

    This function calculates the time derivatives of the
    Lagrange f and g coefficients.

    mu - the gravitational parameter(kmˆ3 / sˆ2)
    a - reciprocal of the semimajor axis(1 / km)
    ro - the radial position at time t(km)
    t - the time elapsed since initial state vector(s)
    r - the radial position after time t(km)
    x - the universal anomaly after time t(kmˆ0.5)
    fDot - time derivative of the Lagrange f coefficient(1 / s)
    gDot - time derivative of the Lagrange g coefficient
    (dimensionless)

    User M - functions required : stumpC, stumpS
    ------------------------------------------------------------
    """

    z = ua * x * x
    # //%...Equation 3.66c: 3.66d :
    fdot = sqrt(mu) / r / ro * (z * stumpS(z) - 1) * x
    gdot = 1. - x * x / r * stumpC(z)
    return fdot, gdot

@njit
def propagate_lagrangian(r0, v0, tof, mu=1.):
    """
    ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
	function[r, v] = rv_from_r0v0(r0, v0, t)
    ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
    This function computes the state vector(r, v) from the
    initial state vector(r0, v0) and the elapsed time.

    mu - gravitational parameter(km^3 / s^2)
    r0 - initial position vector(km)
    v0 - initial velocity vector(km / s)
    t - elapsed time(s)
    r - final position vector(km)
    v - final velocity vector(km / s)

    User M - functions required : kepler_U, f_and_g, fDot_and_gDot
    ------------------------------------------------------------
    """

    #  //%...Magnitudes of r0 and v0 :
    r0_norm = norm(r0)
    v0_norm = norm(v0)

    #//%...Initial radial velocity :
    vr0 = np.dot(r0, v0) / r0_norm
    #//%...Reciprocal of the semimajor axis(from the energy equation) :
    alpha = 2 / r0_norm - v0_norm*v0_norm / mu

    # //%...Compute the universal anomaly :
    x = kepler_U(mu, tof, r0_norm, vr0, alpha)
    # //%...Compute the f and g functions, derivatives of f and g
    f, g = lagrangefg_kepU(mu, x, tof, r0_norm, alpha)
    # //%...Compute the final position vector
    r = f * r0 + g * v0
    # //%...Compute the magnitude of r :
    r_norm = norm(r)
    # //%...Compute the derivatives of f and g
    fdot, gdot = lagrangefDotgDot_kepU(mu, x, r_norm, r0_norm, alpha)
    # //%...Compute the final velocity:
    v = fdot*r0 + gdot*v0

    return r, v


def par2ic(coe, mu = 1.):
    """
    Description:
    
    Trasform the classical orbital element set into 
    position / velocity vectors


    Date:                  20 / 09 / 2013
        Revision : 2
        FIXED r_magnitude = r(not p!)
        FIXED v_z sign(now is correct!)
        Tested by : ----------


    Usage : [r, v] = coe2rvECI(COE, mu)

    Inputs :
                COE : [6]   Classical orbital elements
    [a; e; ainc; gom; pom; anu];
                mu:[1]  grav.parameter of the primary body

        ***       "COE(1)" and "mu" units must be COERENT  ***

    Outputs :
                r : [3x1]    S / C position in ECI
                v : [3x1]    S / C velocity in ECI

    """
    r = [0, 0, 0]
    v = [0, 0, 0]

    a = coe[0]
    e = coe[1]
    ainc = coe[2]
    gom = coe[3]
    pom = coe[4]
    anu = coe[5]

    p = a*(1. - pow(e,2))
    u = pom + anu
    r_norm = p / (1 + e*cos(anu))

    r[0] = r_norm*(cos(gom)*cos(u) - sin(gom)*cos(ainc)*sin(u))
    r[1] = r_norm*(sin(gom)*cos(u) + cos(gom)*cos(ainc)*sin(u))
    r[2] = r_norm*(sin(ainc)*sin(u))

    v[0] = -sqrt(mu / p)* (cos(gom)*(sin(u) + e*sin(pom)) + sin(gom)*cos(ainc)* (cos(u) + e*cos(pom)))
    v[1] = -sqrt(mu / p)* (sin(gom)*(sin(u) + e*sin(pom)) - cos(gom)*cos(ainc)* (cos(u) + e*cos(pom)))
    v[2] = -sqrt(mu / p)* (-sin(ainc)*(cos(u) + e*cos(pom)))  

    return np.array(r), np.array(v)


def ic2par(r, v, mu = 1.0):
    """
    Description:

    Trasform position / velocity vector into
    the classical orbital elements


    Date:                  20 / 09 / 2013
        Revision : 1
        Tested by : ----------


    Usage : COE = rvECI2coe(r, v, mu)

    Inputs :
                r : [3x1]    S / C position in ECI
                v : [3x1]    S / C velocity in ECI
                mu : [1]      grav.parameter of the primary body

        ***  "r", "v", and "mu" units must be COERENT  ***

    Outputs :
                COE : [6]      Classical orbital elements
    [a; e; ainc; gom; pom; anu];
    """

    vect_rDUM = r # np.array(r)
    vect_vDUM = v # np.array(v)

    #Non dimensional units
    rconv = norm(vect_rDUM)
    vconv = sqrt(mu / rconv)

    #working with non - dimensional variables
    vect_r = vect_rDUM / rconv
    vect_v = vect_vDUM / vconv

    r_norm = norm(vect_r)
    vers_i = vect_r / r_norm
    vect_h = cross(vect_r, vect_v)

    vect_e = cross(vect_v, vect_h) - vers_i
    vect_n = cross(np.array([0, 0, 1]), vect_h)
    an = norm(vect_n)

    #_______SEMI-MAJOR AXIS_______
    av2 = dot(vect_v, vect_v)
    a = -0.5 / (av2*0.5 - 1 / r_norm)

    #_______ECCENTRICITY_______
    acca2 = dot(vect_h, vect_h)
    e2 = 1 - acca2 / a

    if (e2<-1e-8):
        e = -1
    elif (e2<1e-8):
        e = 0
    else:
        e = sqrt(e2)

    #_____ORBITAL INCLINATION______
    ainc = arccos(vect_h[2] / sqrt(acca2))

    vers_h = vect_h / sqrt(acca2)
    if (ainc < 1e-6):
        vers_n = np.array([1, 0, 0])
    else:
        vers_n = vect_n / an

    if (e2 < 1e-6):
        vers_e = vers_n
    else:
        vers_e = vect_e / e

    vers_p = cross(vers_h, vers_e)

    #_____RIGHT ASCENSION OF THE ASCENDING NODE______
    gom = arctan2(vers_n[1], vers_n[0])

    #_____ARGUMENT OF PERIAPSIS______
    pom = arctan2(dot(-vers_p,vers_n) , dot(vers_e,vers_n))

    #_____TRUE ANOMALY______
    anu = arctan2(dot(vers_i,vers_p) , dot(vers_i,vers_e))

    #Restore dimensional units
    coe = [a*rconv, e, ainc, gom, pom, anu]

    return coe


def par2eq(coe):
    """
    Description:
    
    Trasform the classical orbital element set into the modified
    equinotial element set

    Inputs :
                COE : [6]   Classical orbital elements
                [a; e; ainc; gom; pom; anu];

    Outputs :
                MEE : [6]   modifies equinotial elements
                [p; f; g; h; k; L];

    """

    #COE
    a = coe[0]
    e = coe[1]
    ainc = coe[2]
    gom = coe[3]
    pom = coe[4]
    anu = coe[5]

    #MEE
    p = a*(1. - e**2)
    f = e*cos(pom + gom)
    g = e*sin(pom + gom)
    h = tan(ainc/2.)*cos(gom)
    k = tan(ainc/2.)*sin(gom)
    L = gom + pom + anu
    mee = [p, f, g, h, k, L]

    return mee


def eq2par(mee):
    """
    Description:
    
    Trasform the modified equinotial element set into the 
    classical orbital element set

    Inputs :
                MEE : [6]   modifies equinotial elements
                [p; f; g; h; k; L];

    Outputs :
                COE : [6]   Classical orbital elements
                [a; e; ainc; gom; pom; anu];
   

    """

    #MEE
    p = mee[0]
    f = mee[1]
    g = mee[2]
    h = mee[3]
    k = mee[4]
    L = mee[5]

    #COE
    a = p/(1 - f**2 - g**2)
    e = sqrt(f**2 + g**2)
    ainc = 2*arctan(sqrt(h**2 + f**2))
    gom = arctan2(k, h)
    pom = arctan2(g*h - f*k, f*h + g*k)
    anu = arctan2(h*sin(L) - k*cos(L), h*cos(L) + k*sin(L)) - pom
    coe = [a, e, ainc, gom, pom, anu]

    return coe


def ic2eq(r, v, mu = 1.0):
    """
    Description:

    Trasform position / velocity vectors into
    the modified equinotial elements

    Inputs :
                r : [3x1]    S / C position in ECI
                v : [3x1]    S / C velocity in ECI
                mu : [1]      grav.parameter of the primary body

        ***  "r", "v", and "mu" units must be COERENT  ***

    Outputs :
                MEE : [6]   modifies equinotial elements
                [p; f; g; h; k; L];
    """

    coe = ic2par(r, v, mu)
    mee = par2eq(coe)

    return mee


def eq2ic(mee, mu = 1.0):
    """
    Description:

    Trasform the modified equinotial elements
    into position / velocity vectors

    Inputs :
                MEE : [6]   modifies equinotial elements
                [p; f; g; h; k; L];
                mu : [1]      grav.parameter of the primary body

        ***  "MEE" and "mu" units must be COERENT  ***

    Outputs :
                r : [3x1]    S / C position in ECI
                v : [3x1]    S / C velocity in ECI
    """

    coe = eq2par(mee)
    r, v = par2ic(coe, mu)

    return r, v


def TBP_eqs(t, s, mu, f, ueq):
    """
    Right-hand side of the system of equations of motion of a spacecraft
    in the two-body problem with low thrust terms.

    input:
    t - (float) time (s)
    s - (np.array) spacecraft state (position, velocity, mass) (km, km/s, kg)
    mu - (float) gravitational constant of the central body (km^3/s^2)
    f - (np.array) spacecraft thrust (kN)
    ueq - (float) spacecraft equivalent ejection velocity (s)

    output:
    s_dot - (np.array) derivatives of the spacecraft state (km/s, km/s^2, kg/s)
    ------------------------------------------------------------
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
    fx = f[0]
    fy = f[1]
    fz = f[2]

    #Auxiliary variables
    r = sqrt(x**2 + y**2 + z**2)

    #Equations of motion
    x_dot = vx
    y_dot = vy
    z_dot = vz
    vx_dot = - mu*x/r**3 + fx/m
    vy_dot = - mu*y/r**3 + fy/m
    vz_dot = - mu*z/r**3 + fz/m
    m_dot = - norm(f)/ueq

    s_dot = np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot, m_dot], dtype='object')

    return s_dot


def TBP_eqs_sph(t, s, mu, f, ueq):
    """
    Right-hand side of the system of equations of motion of a spacecraft
    in the two-body problem with low thrust terms.

    input:
    t - (float) time (s)
    s - (np.array) spacecraft state (position, velocity, mass) (km, km/s, kg)
    mu - (float) gravitational constant of the central body (km^3/s^2)
    f - (np.array) spacecraft thrust (kN)
    ueq - (float) spacecraft equivalent ejection velocity (s)

    output:
    s_dot - (np.array) derivatives of the spacecraft state (km/s, km/s^2, kg/s)
    ------------------------------------------------------------
    """

    #State variables

    #spacecraft position
    r = s[0]
    theta = s[1]
    phi = s[2]

    #spacecraft velocity
    vr = s[3]
    vt = s[4]
    vn = s[5]

    #spacecraft mass
    m = s[6]

    #Control
    fr = f[0]
    ft = f[1]
    fn = f[2]

    #Equations of motion
    r_dot = vr
    theta_dot = vt/(r*cos(phi))
    phi_dot = vn/r
    vr_dot = (vt**2 + vn**2)/r - mu/r**2 + fr/m
    vt_dot = - vr*vt/r + vt*vn*tan(phi)/r + ft/m
    vn_dot = - vr*vn/r - (vt**2)*tan(phi)/r + fn/m
    m_dot = - norm(f)/ueq

    s_dot = np.array([r_dot, theta_dot, phi_dot, vr_dot, vt_dot, vn_dot, m_dot], dtype='object')

    return s_dot


""" Edelbaum criterion for low-thrust orbit transfers """
def Edelbaum(a0, af, ainc0, aincf, amu, mi, T, ueq):

    Di = abs(aincf - ainc0)
    v0 = sqrt(amu/a0)
    vf = sqrt(amu/af)
    a_avg = (a0 + af)/2.
    v_avg = sqrt(amu/a_avg)

    Dv = sqrt(v0**2 - 2.*vf*v0*cos(0.5*np.pi*Di) + vf**2)
    mp = mi*(1. - np.exp(-Dv/ueq))
    Dt = mp/(T/ueq)
    Dtheta = v_avg/a_avg*Dt

    return Dv, mp, Dt, Dtheta


""" Edelbaum criterion for low-thrust rendezvous maneuvers """
def Edelbaum_rendezvous(a0, af, ainc0, aincf, theta0, thetaf, amu, mi, T, ueq):

    #Orbit transfer
    Dv_t, _, _, Dtheta_t = Edelbaum(a0, af, ainc0, aincf, amu, mi, T, ueq)

    #Rendezvous
    thetaf_t = (theta0 + Dtheta_t + 2.*np.pi) % (2.*np.pi)
    a_avg = (a0 + af)/2.
    v_avg = sqrt(amu/a_avg)
    Dtheta_r = abs(thetaf - thetaf_t)
    K = 4./3.*Dtheta_r/Dtheta_t*v_avg/Dv_t
    Dv_r = Dv_t*((sqrt(1. + 4.*K) - 1.)/2.)

    Dv = Dv_t + Dv_r
    mp = mi*(1. - np.exp(-Dv/ueq))
    Dt = mp/(T/ueq)
   
    return Dv, mp, Dt