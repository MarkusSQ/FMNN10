import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp

from project2 import twopBVP, norm

alpha = 0
beta = -1
N = 100
L = np.pi*0.5

outgrid = linspace(0, L, N+2) #The whole interval with end points
ingrid = linspace(1/N,L -1/(L-1),N) #Without the endpoints. 1/N is the first element not 0. L - 1/(L-1) is the next to last element
fvec = np.sin(ingrid) #Take the sine of each vector element
y = twopBVP(fvec, alpha, beta, L, N) #y-vector
sol = -np.sin(outgrid)

# plotting the error as a function of stepsize delx and comparing it to delx^2
delx_range = 1000 
errors = [] #array with all the errors computed with rms norm
stepsizes = [] # array with all the stepsizes
control = [] #array with the stepsize delx^2 to compare to the error
for k in range(2,delx_range): 
   delx = L/(k+1)
   stepsizes.append(delx) #plotting against stepsize, not number of steps
   f = np.sin(np.linspace(1/k, L-1/k, num=k)) #Does not include endpoints
   t = np.linspace(0, L, num=k+2) #Includes endpoints
   sol = -np.sin(t)
   errors.append(norm(twopBVP(f, alpha, beta, L, k) - sol)) #Approximation - exact = error
   control.append(pow(delx,2)) #To compare with the error (to make sure it is of correct order)

plt.loglog(stepsizes, errors, label="error") 
plt.loglog(stepsizes, control, label="delx^2")
plt.legend()
plt.show()