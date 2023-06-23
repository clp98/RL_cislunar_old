#Reference frames for the 4 body problem (taken from the DART paper)

import numpy as np
from numpy import sqrt, cos, sin
from numpy.linalg import inv
from environment.pyKepler import *
from numba import njit


""" RVN to ICRS """
# @njit
def RVNtoICRS(v, vec_rvn):
    """
    :return vec_icrs: vec in ICRS frame

    """
    
    vers_v = v/norm(v)
    vec_r = cross(vers_v, np.array([0., 0., 1.], dtype=np.float64))
    vers_r = vec_r/max(norm(vec_r), 1e-05)
    vers_n = cross(vers_r, vers_v)
    
    vec_icrs = vec_rvn[0]*vers_r + \
        vec_rvn[1]*vers_v + \
        vec_rvn[2]*vers_n

    return vec_icrs


""" RVN to ICRS """
# @njit
def ICRStoRVN(v, vec_icrs):
    """
    :return vec_rvn: vec in RVN frame

    """
    
    vers_v = v/norm(v)
    vec_r = cross(vers_v, np.array([0., 0., 1.], dtype=np.float64))
    vers_r = vec_r/max(norm(vec_r), 1e-05) 
    vers_n = cross(vers_r, vers_v)

    vec_rvn = np.array([dot(vec_icrs, vers_r), dot(vec_icrs, vers_v), dot(vec_icrs, vers_n)], dtype=np.float64)

    return vec_rvn


# @njit
def ICRS2RTN(vec_ICRS, COE_body):
    """
    To pass from ICRS to RTN (the two must share the same origin!).
    All the arguments (and outputs) must be non-dimensional!
    RTN: radial-transverse-normal second-third body. It pulses in order to keep r unity.
    """

    # Read COE
    a = COE_body[0]
    e = COE_body[1]
    inc = COE_body[2]
    OMEGA = COE_body[3]
    omega = COE_body[4]
    anu = COE_body[5]

    p = a * (1. - e**2)
    r = p / (1. + e * cos(anu))

    # Rotation matrix
    C_IE = []
    C_IE.append([cos(omega + anu) * cos(OMEGA) - sin(omega + anu) * cos(inc) * sin(OMEGA), \
    	cos(omega + anu) * sin(OMEGA) + sin(omega + anu) * cos(inc) * cos(OMEGA), \
    	sin(omega + anu) * sin(inc)])
    C_IE.append([-sin(omega + anu) * cos(OMEGA) - cos(omega + anu) * cos(inc) * sin(OMEGA), \
    	-sin(omega + anu) * sin(OMEGA) + cos(omega + anu) * cos(inc) * cos(OMEGA), \
    	cos(omega + anu) * sin(inc)])
    C_IE.append([sin(inc) * sin(OMEGA), -sin(inc) * cos(OMEGA), cos(inc)])
    C_IE = np.array(C_IE)
    C_IN = (1. / r) * C_IE

    # Output
    vec_RTN = C_IN @ vec_ICRS

    return vec_RTN


def rvICRS2RTN(r_ICRS, v_ICRS, mu_sys, COE_body):
    """
    To pass from ICRS to RTN (the two must share the same origin!).
    All the arguments (and outputs) must be non-dimensional!
    RTN: radial-transverse-normal second-third body. It pulses in order to keep r unity.
    """

    # Read COE
    a = COE_body[0]
    e = COE_body[1]
    inc = COE_body[2]
    OMEGA = COE_body[3]
    omega = COE_body[4]
    anu = COE_body[5]

    p = a * (1. - e**2)
    h = sqrt(p * mu_sys)
    r = p / (1. + e * cos(anu))
    ni_dot = h / (r**2)
    eta1 = (e * sin(anu)) / (1. + e * cos(anu))

    # Rotation matrix
    C_IE = []
    C_IE.append([cos(omega + anu) * cos(OMEGA) - sin(omega + anu) * cos(inc) * sin(OMEGA), \
    	cos(omega + anu) * sin(OMEGA) + sin(omega + anu) * cos(inc) * cos(OMEGA), \
    	sin(omega + anu) * sin(inc)])
    C_IE.append([-sin(omega + anu) * cos(OMEGA) - cos(omega + anu) * cos(inc) * sin(OMEGA), \
    	-sin(omega + anu) * sin(OMEGA) + cos(omega + anu) * cos(inc) * cos(OMEGA), \
    	cos(omega + anu) * sin(inc)])
    C_IE.append([sin(inc) * sin(OMEGA), -sin(inc) * cos(OMEGA), cos(inc)])
    C_IE = np.array(C_IE)
    C_IN = (1. / r) * C_IE

    eye = np.eye(3)

    C_IEperC_EI_dot = []
    C_IEperC_EI_dot.append([0., -1., 0.])
    C_IEperC_EI_dot.append([1., 0., 0.])
    C_IEperC_EI_dot.append([0., 0., 0.])
    C_IEperC_EI_dot = ni_dot * np.array(C_IEperC_EI_dot)

    C_INperC_NI_dot = C_IEperC_EI_dot + ni_dot * eta1 * eye

    # Output
    r_RTN = C_IN @ r_ICRS
    v_RTN = C_IN @ v_ICRS - C_INperC_NI_dot @ r_RTN

    return r_RTN, v_RTN


# @njit
def RTN2ICRS(vec_RTN, COE_body):
    """
    To pass from RTN to ICRS (the two must share the same origin!).
    All the arguments (and outputs) must be non-dimensional.
    """

    # Read COE
    a = COE_body[0]
    e = COE_body[1]
    inc = COE_body[2]
    OMEGA = COE_body[3]
    omega = COE_body[4]
    anu = COE_body[5]

    p = a * (1 - e**2)
    r = p / (1 + e * cos(anu))

    # Rotation matrix
    C_IE = []
    C_IE.append([cos(omega + anu) * cos(OMEGA) - sin(omega + anu) * cos(inc) * sin(OMEGA), \
    	cos(omega + anu) * sin(OMEGA) + sin(omega + anu) * cos(inc) * cos(OMEGA), \
    	sin(omega + anu) * sin(inc)])
    C_IE.append([-sin(omega + anu) * cos(OMEGA) - cos(omega + anu) * cos(inc) * sin(OMEGA), \
    	-sin(omega + anu) * sin(OMEGA) + cos(omega + anu) * cos(inc) * cos(OMEGA), \
    	cos(omega + anu) * sin(inc)])
    C_IE.append([sin(inc) * sin(OMEGA), -sin(inc) * cos(OMEGA), cos(inc)])
    C_IE = np.array(C_IE)
    C_IN = (1. / r) * C_IE

    # Output
    vec_ICRS = inv(C_IN) @ vec_RTN

    return vec_ICRS


# @njit
def rvRTN2ICRS(r_RTN, v_RTN, mu_sys, COE_body):
    """
    To pass from RTN to ICRS (the two must share the same origin!).
    All the arguments (and outputs) must be non-dimensional.
    """

    # Read COE
    a = COE_body[0]
    e = COE_body[1]
    inc = COE_body[2]
    OMEGA = COE_body[3]
    omega = COE_body[4]
    anu = COE_body[5]

    p = a * (1 - e**2)
    h = sqrt(p * mu_sys)
    r = p / (1 + e * cos(anu))
    ni_dot = h / r**2
    eta1 = (e * sin(anu)) / (1 + e * cos(anu))

    # Rotation matrix
    C_IE = []
    C_IE.append([cos(omega + anu) * cos(OMEGA) - sin(omega + anu) * cos(inc) * sin(OMEGA), \
    	cos(omega + anu) * sin(OMEGA) + sin(omega + anu) * cos(inc) * cos(OMEGA), \
    	sin(omega + anu) * sin(inc)])
    C_IE.append([-sin(omega + anu) * cos(OMEGA) - cos(omega + anu) * cos(inc) * sin(OMEGA), \
    	-sin(omega + anu) * sin(OMEGA) + cos(omega + anu) * cos(inc) * cos(OMEGA), \
    	cos(omega + anu) * sin(inc)])
    C_IE.append([sin(inc) * sin(OMEGA), -sin(inc) * cos(OMEGA), cos(inc)])
    C_IE = np.array(C_IE)
    C_IN = (1. / r) * C_IE

    eye = np.eye(3)

    C_IEperC_EI_dot = []
    C_IEperC_EI_dot.append([0., -1., 0.])
    C_IEperC_EI_dot.append([1., 0., 0.])
    C_IEperC_EI_dot.append([0., 0., 0.])
    C_IEperC_EI_dot = ni_dot * np.array(C_IEperC_EI_dot)

    C_INperC_NI_dot = C_IEperC_EI_dot + ni_dot * eta1 * eye

    # Output
    r_ICRS = inv(C_IN) @ r_RTN
    v_ICRS = inv(C_IN) @ v_RTN + inv(C_IN) @ C_INperC_NI_dot @ r_RTN

    return r_ICRS, v_ICRS


def ICRS2ORB(vec_ICRS, COE_body):
    """
    To pass from ICRS to ORB (the two must share the same origin!).
    All the arguments (and outputs) must be non-dimensional!
    ORB: perifocal frame second-third body.
    """

    # Read COE
    inc = COE_body[2]
    OMEGA = COE_body[3]
    omega = COE_body[4]

    # Rotation matrix
    C_IE = []
    C_IE.append([cos(omega) * cos(OMEGA) - sin(omega) * cos(inc) * sin(OMEGA), \
    	cos(omega) * sin(OMEGA) + sin(omega) * cos(inc) * cos(OMEGA), \
    	sin(omega) * sin(inc)])
    C_IE.append([-sin(omega) * cos(OMEGA) - cos(omega) * cos(inc) * sin(OMEGA), \
    	-sin(omega) * sin(OMEGA) + cos(omega) * cos(inc) * cos(OMEGA), \
    	cos(omega) * sin(inc)])
    C_IE.append([sin(inc) * sin(OMEGA), -sin(inc) * cos(OMEGA), cos(inc)])
    C_IE = np.array(C_IE)

    # Output
    vec_ORB = C_IE @ vec_ICRS

    return vec_ORB


def rvICRS2ORB(r_ICRS, v_ICRS, COE_body):
    """
    To pass from ICRS to ORB (the two must share the same origin!).
    All the arguments (and outputs) must be non-dimensional!
    ORB: perifocal frame second-third body.
    """

    # Read COE
    inc = COE_body[2]
    OMEGA = COE_body[3]
    omega = COE_body[4]

    # Rotation matrix
    C_IE = []
    C_IE.append([cos(omega) * cos(OMEGA) - sin(omega) * cos(inc) * sin(OMEGA), \
    	cos(omega) * sin(OMEGA) + sin(omega) * cos(inc) * cos(OMEGA), \
    	sin(omega) * sin(inc)])
    C_IE.append([-sin(omega) * cos(OMEGA) - cos(omega) * cos(inc) * sin(OMEGA), \
    	-sin(omega) * sin(OMEGA) + cos(omega) * cos(inc) * cos(OMEGA), \
    	cos(omega) * sin(inc)])
    C_IE.append([sin(inc) * sin(OMEGA), -sin(inc) * cos(OMEGA), cos(inc)])
    C_IE = np.array(C_IE)

    # Output
    r_ORB = C_IE @ r_ICRS
    v_ORB = C_IE @ v_ICRS

    return r_ORB, v_ORB


def ORB2ICRS(vec_ORB, COE_body):
    """
    To pass from ORB to ICRS (the two must share the same origin!).
    All the arguments (and outputs) must be non-dimensional.
    ORB: perifocal frame second-third body.
    """

    # Read COE
    inc = COE_body[2]
    OMEGA = COE_body[3]
    omega = COE_body[4]

    # Rotation matrix
    C_IE = []
    C_IE.append([cos(omega) * cos(OMEGA) - sin(omega) * cos(inc) * sin(OMEGA), \
    	cos(omega) * sin(OMEGA) + sin(omega) * cos(inc) * cos(OMEGA), \
    	sin(omega) * sin(inc)])
    C_IE.append([-sin(omega) * cos(OMEGA) - cos(omega) * cos(inc) * sin(OMEGA), \
    	-sin(omega) * sin(OMEGA) + cos(omega) * cos(inc) * cos(OMEGA), \
    	cos(omega) * sin(inc)])
    C_IE.append([sin(inc) * sin(OMEGA), -sin(inc) * cos(OMEGA), cos(inc)])
    C_IE = np.array(C_IE)

    # Output
    vec_ICRS = inv(C_IE) @ vec_ORB

    return vec_ICRS


def rvORB2ICRS(r_ORB, v_ORB, COE_body):
    """
    To pass from ORB to ICRS (the two must share the same origin!).
    All the arguments (and outputs) must be non-dimensional.
    ORB: perifocal frame second-third body.
    """

    # Read COE
    inc = COE_body[2]
    OMEGA = COE_body[3]
    omega = COE_body[4]

    # Rotation matrix
    C_IE = []
    C_IE.append([cos(omega) * cos(OMEGA) - sin(omega) * cos(inc) * sin(OMEGA), \
    	cos(omega) * sin(OMEGA) + sin(omega) * cos(inc) * cos(OMEGA), \
    	sin(omega) * sin(inc)])
    C_IE.append([-sin(omega) * cos(OMEGA) - cos(omega) * cos(inc) * sin(OMEGA), \
    	-sin(omega) * sin(OMEGA) + cos(omega) * cos(inc) * cos(OMEGA), \
    	cos(omega) * sin(inc)])
    C_IE.append([sin(inc) * sin(OMEGA), -sin(inc) * cos(OMEGA), cos(inc)])
    C_IE = np.array(C_IE)

    # Output
    r_ICRS = inv(C_IE) @ r_ORB
    v_ICRS = inv(C_IE) @ v_ORB

    return r_ICRS, v_ICRS


def rotation_around_axis(axis, angle, vector):

    C = []
    if axis == 1:
        C.append([1., 0., 0.])
        C.append([0., cos(angle), -sin(angle)])
        C.append([0., sin(angle), cos(angle)])
    elif axis == 2:
        C.append([cos(angle), 0., sin(angle)])
        C.append([0., 1., 0.])
        C.append([-sin(angle), 0., cos(angle)])
    else:
        C.append([cos(angle), -sin(angle), 0.])
        C.append([sin(angle), cos(angle), 0.])
        C.append([0., 0., 1.])
    C = np.array(C)

    rotated_vector = C @ vector

    return rotated_vector


def rotation_matrix_Euler_angles(phi, theta, psi):
    """ 
    Rotation matrix from Euler angles
    """
    Cphi = cos(phi)
    Sphi = sin(phi)
    Ctheta = cos(theta)
    Stheta = sin(theta)
    Cpsi = cos(psi)
    Spsi = sin(psi)

    A = np.array([[Ctheta*Cpsi, Ctheta*Spsi, -Stheta], \
        [Sphi*Stheta*Cpsi - Cphi*Spsi, Sphi*Stheta*Spsi + Cphi*Cpsi, Sphi*Ctheta], \
        [Cphi*Stheta*Cpsi + Sphi*Spsi, Cphi*Stheta*Spsi - Sphi*Cpsi, Cphi*Ctheta]], dtype=np.float64)
    
    return A

def rotation_matrix_small_angles(phi, theta, psi):
    """ 
    Rotation matrix under the small-angle assumption
    """

    A = np.array([[1., psi, -theta], \
        [-psi, 1., phi], \
        [theta, -phi, 1.]], dtype=np.float64)
    
    return A


def Euler_angles_from_rotation_matrix(A):
    """
    Euler angles, starting from the rotation matrix
    """

    phi = np.arctan2(A[1][2], A[2][2])
    theta = np.arcsin(-A[0][2])
    psi = np.arctan2(A[0][1], A[0][0])

    return phi, theta, psi