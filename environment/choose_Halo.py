#this script chooses randomly a set of initial conditions
from random import randint
import numpy as np       
     
def choose_Halo(filename, single_matrix):
    with open(filename, 'r') as f:
        lines=f.readlines()
        rv_matrix=[]
        for line in lines:
            rv_matrix.append(line.split(' '))

    k=randint(0,100)
    i=randint(1,3)  #select a random matrix 
    if single_matrix:  #extract only from the first matrix (Halo-1)
        r0=np.array(rv_matrix[k][0:3])  #initial position
        v0=np.array(rv_matrix[k][3:6])  #initial velocity

    else:  #extract from any of the matrices (Halo-j)
        j=100*i+k
        r0=np.array(rv_matrix[j][0:3])  #initial position
        v0=np.array(rv_matrix[j][3:6])  #initial velocity
        
    return r0, v0

