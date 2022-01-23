import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve.linsolve import spsolve

# PART 1
# TASK 1.1
def eulerstep(A, uold, dt): #Task 1.1
  unew = uold + dt*np.matmul(A,uold)
  return unew

def TRstep(Tdx, uold, dt):
    I = np.identity(len(Tdx))
    unew = np.linalg.solve((I - 0.5*dt*Tdx),np.matmul((I + 0.5*dt*Tdx),uold)) #Trapezoidal instead of forward euler
    np.linalg.solve
    return unew

def diffusionSolver(tend, g, N, M):
    dx = 1/(N+1)
    dt = tend/M #tend is the end value of time

    # creating the toeplitz matrix NxN-matrix
    row = np.zeros(N)
    row[0] = -2/(dx**2)
    row[1] = 1/(dx**2)
    t_matrix = lg.toeplitz(row, row)

    # the two grids
    xx = np.linspace(0, 1, num=N+2)
    tt = np.linspace(0, tend, num=M+1)

    # time discretization with Eulers method
    uold = g(np.linspace(0+dx,1-dx,num=N)) # without end points
    u_values = np.zeros((N+2,M+1)) # empty matrix to put the functions in (compare with results in eulerint)
    for t in range(M+1):
        u_values[:,t] = np.array([0,*uold,0]) # inserted as a column, boundaries are added to the array
        #uold = eulerstep(t_matrix, uold, dt) #Task 1.1
        uold = TRstep(t_matrix, uold, dt) #task 1.2    
    
    # creating the grids and plot
    [T, X] = np.meshgrid(tt,xx)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.plot(X, T, u_values, label='parametric curve')
    ax.plot_wireframe(X,T,u_values)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('$u$')

    plt.title('Trapezoidal rule instead of forward Euler (M=100)') #Trapezoidal rule is A-stable. We therefore avoid the CFL condition and no longer have a severe restriction on the time step
    #ax.yaxis._axinfo['label']['space_factor'] = 3.0

    plt.show()

def g(x): # initial value function
  return 10*(0.5-np.absolute(x-0.5))

# Lägsta värdet då N=30 på M var 190. dx = 1/30 och dt = 0.1/190. M=187 var ett bra värde för
# att se hur CFL-violation ser ut. CFL för Expl Euler är dt/(dx**2) leq 1/2. med M=187 har vi CFL= 0.514
diffusionSolver(0.1, g, 30, 100)
