import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp

from project2 import twopBVP, norm

alpha = 0
beta = 0
L = 10 #m
E = 1.9*pow(10,11) #N/(m^2)
q = -50*pow(10,3)
N = 999

outgrid = linspace(0, L, N+2) #The whole interval with end points
ingrid = linspace(1/N,L -1/(L-1),N) #Only has internal points

def I(x,L):
    return pow(10,-3)*(3-2*pow(np.cos(np.pi*x/L),12))

fvec = I(ingrid,L)

M = twopBVP(q,alpha,beta,L,N-2) #Two grid points added each time
u = twopBVP(M/(E*fvec), alpha, beta, L, N)

print(u[500]) #For testing purposes

plt.plot(outgrid,u)
plt.show()
print()