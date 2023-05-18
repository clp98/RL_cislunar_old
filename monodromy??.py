from math import *
from environment.BER4BP import *
from environment.CR3BP import *

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



g =  6.67430e-2
m_1 = 5.972e24
m_2 = 7.347e22
r_12 = 384.4e6
m = m_1+m_2
mu = g*m

t = (2*np.pi*r_12**1.5)/(sqrt(mu))

print(t)



