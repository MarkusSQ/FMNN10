import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp

from SturmSolver import sturmSolver

alpha = 0
beta = 0
N = 499
L = 1

outgrid = linspace(0, L, num=N+2) #The whole interval with end points
for i in range(3):
   plt.plot(outgrid, sturmSolver(alpha, beta, L, N)[1][i], label=i)
plt.show()