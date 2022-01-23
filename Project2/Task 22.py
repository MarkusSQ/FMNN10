import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
from scipy.linalg.special_matrices import toeplitz
import scipy.sparse as sp

from SchroedingerSolver import schroedingerSolver

alpha = 0
beta = 0
N = 499
L = 1

def potential0(x):
   return 0*x

def potential1(x):
   return 700*(0.5-abs(x-0.5))

def potential2(x):
   return 800*(np.sin(np.pi*x)**2)

def potential3(x):
   return np.heaviside(x-0.3,1)-np.heaviside(x-0.4,1)+np.heaviside(x-0.65,1)+np.heaviside(x-0.73,1)

outgrid = linspace(0, L, num=N+2) #The whole interval with end points
for i in range(6):
  plt.plot(outgrid, schroedingerSolver(alpha, beta, L, N+1, potential0)[2][i], label=i) #V(x) = 0
plt.show()

#V(x) = 700(0.5- abs(x-0.5))
# for i in range(6):
#   plt.plot(outgrid, schroedingerSolver(alpha, beta, L, N+1, potential1)[2][i], label=i) #V(x) = 0
# plt.show()

# for i in range(6):
#    plt.plot(outgrid, schroedingerSolver(alpha, beta, L, N+1, potential2)[2][i], label=i) #V(x) = 0
# plt.show()

# for i in range(6):
#    plt.plot(outgrid, schroedingerSolver(alpha, beta, L, N+1, potential3)[2][i], label=i) #V(x) = 0
# plt.show()

#print(toeplitz([3,3,3])-np.diag([3,3,3]))
# T = sp.diags([1, 2, 3], 0) + sp.diags([1,2],1)
# print(T)

#print(sp.diags(2,3,1))
