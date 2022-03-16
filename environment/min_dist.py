import numpy as np
from environment.CR3BP import *
from scipy.optimize import minimize_scalar
from scipy.optimize import Bounds


def dist_state_orbit(t, s, sLy_0):

    sLy = propagate_cr3bp_free(sLy_0, np.array([0., t], dtype=np.float64))

    d = norm(s - sLy[-1])

    return d


def dist_state_orbit_der(t, s, sLy_0):

    sLy = propagate_cr3bp_free(sLy_0, np.array([0., t], dtype=np.float64))

    d = norm(s - sLy[-1])
    d_vec = s - sLy[-1]

    sLy_dot = CR3BP_eqs_free(t, sLy[-1])

    der = np.dot(-d_vec/d, sLy_dot)

    return der


def nearest_point(s, sLy_0, TLy):

    res = minimize_scalar(dist_state_orbit, 0., args=(s, sLy_0), method='bounded',
               bounds=(0., TLy), options={'disp': False})
    
    sLy = propagate_cr3bp_free(sLy_0, np.array([0., res.x], dtype=np.float64))

    return sLy[-1], res.fun