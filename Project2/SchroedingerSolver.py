import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
from scipy.linalg.special_matrices import toeplitz
import scipy.sparse as sp

from project2 import norm

def schroedingerSolver(alpha, beta, L, N, potential):
    delx = L/N

    boundries = np.zeros(N) #A vector with correct dimensions, however only with the boundaries
    boundries[0] = alpha/(delx**2)
    boundries[-1] = beta/(delx**2)

    x = np.linspace(0,L, num = N)
    V_x= potential(x) #Our discretized potential

    row = np.zeros(N) #tridiagonal elements
    row[0] = (-2)/(delx**2)
    row[1] = 1/(delx**2)

    T = lg.toeplitz(row,row) - np.diag(V_x) #Adjusting our toeplitz matrix with our potential

    
    V, F = lg.eig(T) #V for eigenvalues, F for eigenvectors

    ind = np.argsort(abs(V)) #Sorting the eigenvectors
    lamda = V[ind]
    F = F[:,ind]

    eigenVectors = []
    prob_densities = []
    for i in range(6):
        eigenVectors.append(F[:,i])
        eigenVectors[i] = eigenVectors[i] - boundries
        eigenVectors[i] = [alpha, *eigenVectors[i]]
        eigenVectors[i] = eigenVectors[i]/(2*norm(eigenVectors[i])) - lamda[i]*np.ones(N+1) #Factor 2 is chosen for increased visibility


        prob_densities.append(np.absolute(eigenVectors[i])**2)

    return lamda, eigenVectors, prob_densities