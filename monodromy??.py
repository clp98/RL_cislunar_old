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


######################################################################################################################################################################################################################

#put into CR3BP
#different reward approach

def halo_dist_r(t, y, data):

    r = np.array([state['r'][0], state['r'][1], state['r'][2]])
    r_moon = np.array([1 - mu, 0., 0.])  #moon position in relative system
    R_moon = 1738/l_star  #[km]
    distance_r = norm(r-r_moon) - R_moon

    return distance_r



def halo_dist_v(t, y, data):

    v = np.array([state['v'][0], state['v'][1], state['v'][2]])
    r_moon = np.array([1 - mu, 0., 0.])  #moon position in relative system
    V_moon = 1738/v_star  #[km]
    distance_v = 

    return distance_v



#put into next_state
hitMoon.terminal = True
halo_dist_r.terminal = False
halo_dist_v.terminal = False
events = (hitMoon, halo_dist_r, halo_dist_v)
data = np.concatenate((control, self.ueq), axis=None)

#Solve equations of motion with CR3BP
solution_int = solve_ivp(fun=CR3BP_equations_controlled_ivp, t_span=t_span, t_eval=None, y0=s, method='RK45', events=events, \
                args=(data,), rtol=1e-7, atol=1e-7)

r_min = y_events[0][0]
r_min = y_events[0][0]





