#Integration with RK4 

import numpy as np


def rk4_single(f, y0, t0, dt, data):

    n_eq = len(y0)

    f0 = np.zeros(n_eq)
    f(t0, y0, f0, data)
    
    t1 = t0 + dt / 2.0
    y1 = y0 + dt * f0 / 2.0
    f1 = np.zeros(n_eq)
    f(t1, y1, f1, data)
    
    t2 = t0 + dt / 2.0
    y2 = y0 + dt * f1 / 2.0
    f2 = np.zeros(n_eq)
    f(t2, y2, f2, data)
    
    t3 = t0 + dt	
    y3 = y0 + dt * f2
    f3 = np.zeros(n_eq)
    f(t3, y3, f3, data)
    
    y = y0 + dt * (f0 + 2.0 * f1 + 2.0 * f2 + f3) / 6.0

    return y


def rk4_multi(f, y0, t0, tf, n_steps, data):

    dt = (tf - t0) / float(n_steps)
    t0_new = t0
    y0_new = y0

    for _ in range(n_steps):
        y0_new = rk4_single(f, y0_new, t0_new, dt, data)
        t0_new = t0_new + dt

    return y0_new


def rk4_prec(f, y0, t0, tf, rtol, data):

    dt = tf - t0
    n_steps = int(np.ceil(dt/(rtol**(1/4))))

    return rk4_multi(f, y0, t0, tf, n_steps, data)