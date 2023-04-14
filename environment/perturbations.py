#Perturbations

import numpy as np
from numpy.linalg import norm

#solar radiation pressure
if with_srp:
    # Sun-sc distance in AU
    r14 = norm(r14)*rconv/AU

    #SRP at sc distance
    srp_norm_1AU = 4.56*10**-6  #[N/m^2]
    srp_norm = srp_norm_1AU*(1.5*10**8)/(r14**2)  #[N/m^2]
    #r14_dotdot_withsrp=r14_dotdot+srp*self.A_sc/self.m_sc

    #SRP force
    f_srp_norm = srp_norm*self.A_sc*1e-03
    f_srp_x = f_srp_norm*r14[0]/norm(r14)
    f_srp_y = f_srp_norm*r14[1]/norm(r14)
    f_srp_z = f_srp_norm*r14[2]/norm(r14)
    f_srp = np.array([f_srp_x, f_srp_y, f_srp_z], dtype=np.float64)
else:
    f_srp = np.zeros(3, dtype=np.float64)
