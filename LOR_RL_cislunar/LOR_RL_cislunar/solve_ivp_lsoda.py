from numbalsoda import lsoda_sig, lsoda
from numba import njit, cfunc

@njit
def solve_ivp_lsoda(funcptr, s, t_eval, rtol, atol, data = None):

    if data is None:
        sol, success = lsoda(funcptr, s, t_eval, rtol = rtol, atol = atol)
    else:
        sol, success = lsoda(funcptr, s, t_eval, data = data, rtol = rtol, atol = atol)

    return sol, success