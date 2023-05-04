from math import *

U_xx = 4.3809
omega_xy = 1.8626
lambda_1 = 2.1586
lambda_2 = -lambda_1

#Wronskian matrix W(t)
W = [[exp(lambda_1*t), exp(lambda_2*t), cos(omega_xy*t), sin(omega_xy*t)], \
    [lambda_1*exp(lambda_1*t), lambda_2*exp(lambda_2*t), -omega_xy*sin(omega_xy*t), omega_xy*cos(omega_xy*t)], \
    [exp(lambda_1*t)*(lambda_1**2-U_xx)/(2*lambda_1), exp(lambda_2*t)*(lambda_2**2-U_xx)/(2*lambda_2), -sin(omega_xy*t)*(omega_xy**2+U_xx)/(2*omega_xy), cos(omega_xy*t)*(omega_xy**2+U_xx)/(2*omega_xy)], \
    [exp(lambda_1*t)*(lambda_1**2-U_xx)/2, exp(lambda_2*t)*(lambda_2**2-U_xx)/2, -cos(omega_xy*t)*(omega_xy**2+U_xx)/2, -sin(omega_xy*t)*(omega_xy**2+U_xx)/2]]

