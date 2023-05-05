from math import *
from environment.BER4BP import *

#U_xx = 4.3809
#omega_xy = 1.8626
#lambda_1 = 2.1586
#lambda_2 = -lambda_1

#Wronskian matrix W(t)
#W = [[exp(lambda_1*t), exp(lambda_2*t), cos(omega_xy*t), sin(omega_xy*t)], \
#    [lambda_1*exp(lambda_1*t), lambda_2*exp(lambda_2*t), -omega_xy*sin(omega_xy*t), omega_xy*cos(omega_xy*t)], \
#    [exp(lambda_1*t)*(lambda_1**2-U_xx)/(2*lambda_1), exp(lambda_2*t)*(lambda_2**2-U_xx)/(2*lambda_2), -sin(omega_xy*t)*(omega_xy**2+U_xx)/(2*omega_xy), cos(omega_xy*t)*(omega_xy**2+U_xx)/(2*omega_xy)], \
#    [exp(lambda_1*t)*(lambda_1**2-U_xx)/2, exp(lambda_2*t)*(lambda_2**2-U_xx)/2, -cos(omega_xy*t)*(omega_xy**2+U_xx)/2, -sin(omega_xy*t)*(omega_xy**2+U_xx)/2]]




#next_state
halo_dist_r.terminal = True
events = (halo_dist_r)
data = np.concatenate((control, self.ueq), axis=None)

#Solve equations of motion with CR3BP
solution_int = solve_ivp(fun=CR3BP_equations_controlled_ivp, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events, \
                args=(data,), rtol=1e-7, atol=1e-7)





