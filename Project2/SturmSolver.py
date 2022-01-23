import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve.linsolve import spsolve

def sturmSolver(alpha, beta, L, N):
    delx = L/N

    boundries = np.zeros(N) #A vector with correct dimensions, however only with the boundaries
    boundries[0] = alpha/(delx**2)
    boundries[-1] = 2*beta/(delx)

    row = np.zeros(N) #tridiagonal elements
    row[0] = -2/(delx**2)
    row[1] = 1/(delx**2)

    T = lg.toeplitz(row,row)
    T[N-1][N-2] = 2/(delx**2) #Changing to fit our equation
    
    
    V, F = lg.eig(T) #V for eigenvalues, F for eigenvectors

    ind = np.argsort(abs(V)) #Sorting the eigenvectors
    lamda = V[ind]
    F = F[:,ind]

    eigenVectors = [F[:,0],F[:,1],F[:,2]]
    for i in range(3):
        eigenVectors[i] = eigenVectors[i] - boundries
        eigenVectors[i] = [alpha, *eigenVectors[i]]

    return lamda, eigenVectors

#lamba = 2*np.ones(499)
y = sturmSolver(0, 0, 1, 499)
grid = np.linspace(0,1,num=499)
plt.plot(grid,y[1])
plt.show()
