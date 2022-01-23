import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp
import scipy.sparse.linalg
from Part1 import TRstep, g

def convdif(u,a,d,dt,dx):
    N = len(u)
    M = 1/dt
    dx = 1/N
    dx2 = dx**2

    # Creating the Toeplitz matrix
    row = np.zeros(N)
    row[0] = -2/(dx**2)
    row[1] = 1/(dx**2)
    t_matrix = lg.toeplitz(row, row)
    t_matrix[N-1][0] = 1/(dx**2) #Super
    t_matrix[0][N-1] = 1/(dx**2) #sub

    # Creating derivative matrix
    diagonals = [(-0.5/dx)*np.ones(N-1), (1/dx)*np.zeros(N), (1/dx)*np.ones(N-1)]
    s_matrix = scipy.sparse.diags(diagonals, [-1,0,1]).toarray()
    s_matrix[N-1][0] = 1/(2*dx) # Super
    s_matrix[0][N-1] = -1/(2*dx) # sub

    unew = TRstep(d*t_matrix-a*s_matrix,u,dt)

    return unew

def convdiffSolver(N,M,a,d,g):
    Pe = abs(a/d) #Peclét number
    print(Pe)
    
    tend = 1
    dt = tend/M
    dx = 1/N

    print(Pe*dx) #mesh-peclet

    print(d*dt/(dx**2))

    xx = np.linspace(0,1, num = N+1)
    tt = np.linspace(0, tend, num = M+1)

    uold = g(np.linspace(0,1-dx, num=N))
    u_values = np.zeros((N+1,M+1))

    for t in range(M+1):
        u_values[:,t] = np.array([*uold,0])
        uold = convdif(uold, a, d, dt, dx)
    
    [T, X] = np.meshgrid(tt,xx)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X,T,u_values)
    plt.show()

convdiffSolver(10,250,1,0.1,g)