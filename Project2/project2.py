import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp

#Task 1.1

def twopBVP(fvec, alpha, beta, L, N):
    delx = L/(N+1)
    
    boundries = np.zeros(N) #We create a vector with correct dimension, however only with boundary values
    boundries[0] = alpha/(delx**2)
    boundries[-1] = beta/(delx**2)

    row = np.zeros(N) #tridiagonal elements
    row[0] = -2/(delx**2)
    row[1] = 1/(delx**2)
    

    fvec = fvec - boundries #we add the boundaries

    #Solve the equation system, add boundaries as first and last elements
    result = lg.solve_toeplitz((row, row), fvec)
    y = [alpha, *result, beta] # * transposes

    return y

def norm(vector): #root mean square norm
    norm = 0
    for a in range(len(vector)):
        norm = norm + vector[a]**2
    norm  = norm/(len(vector)+1)

    return norm
