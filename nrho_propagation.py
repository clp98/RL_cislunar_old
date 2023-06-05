#NRHO orbit propagation from IC

import os
import sys
import time
import numpy as np
from numpy.linalg import norm
from environment.CR3BP import propagate_cr3bp_free, Jacobi_const, CR3BP_eqs_free_lsoda


mu_NRHO = 0.012150529811497

#Initial conditions relative to L2 point
x0 = -0.167809008979642
y0 = 0.
z0 = 0.0286160249739854
vx0 = 0.
vy0 = 0.882937858803859
vz0 = 0.

#Period of the orbit
Dt = 2.030900008526994

#L2 position
X_L2 = 1.1556


#Create file txt with NRHO orbits
f_out_L2 = open('NRHO_L2_rv.txt', "w")
f_out_L2.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("x", "y", "z", "vx", "vy", "vz", "C", "T"))


N_orbits = 100
N_times = 101

#L2 orbit propagation
for i in range(N_orbits):
    t_span = [0., Dt]
    t_eval = np.linspace(0., Dt, N_times)
    r0 = np.array([x0+X_L2, y0, z0])
    v0 = np.array([vx0, vy0, vz0])

    s0 = np.concatenate((r0, v0), axis=None)
    sol = propagate_cr3bp_free(s0, t_eval=t_eval)
    #sol = CR3BP_eqs_free_lsoda(s0, t_eval=t_eval)
    
    for j in range(N_times):
        C = Jacobi_const(sol[j][0], sol[j][1], sol[j][2], sol[j][3], sol[j][4], sol[j][5])
        f_out_L2.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
                % (sol[j][0], sol[j][1], sol[j][2], sol[j][3], sol[j][4], sol[j][5], C, Dt))
    f_out_L2.write("\n\n")
f_out_L2.close()
