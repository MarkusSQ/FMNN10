import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp
import scipy.sparse.linalg

def LaxWen(u,amu): #Lax-Wendroff Scheme for advection equation
    sub = (amu/2)*(1+amu)
    main = 1-pow(amu,2)
    sup = (amu/2)*(amu-1)

    N = len(u)

    #circ = (np.diag(sub*np.ones(N-1,1),-1) + np.diag(main*np.ones(N,1),0)+np.diag(sup*np.ones(N-1,1)-1))

    diagonals = [sub*np.ones(N-1),main*np.ones(N), sup*np.ones(N-1)]
    circ = scipy.sparse.diags(diagonals,[-1,0,1]).toarray()

    circ[N-1,0] = sup #Since the diagonals wrap around
    circ[0,N-1] = sub

    unew = np.matmul(circ,u)

    return unew



def advectionSolver(tend, g, a, N, M):
    dx = 1/N
    dt = tend/M
    xx = np.linspace(0,1, num = N+1)
    tt = np.linspace(0, tend, num = M+1)

    amu = a*(dt/dx)
    print(amu)

    uold = g(np.linspace(0,1-dx, num=N))
    u_values = np.zeros((N+1,M+1))
    rms_norms = np.zeros(M+1)

    for t in range(M+1):
        rms_norms[t] = RMS(np.array([*uold,0])) #We have with the boundary!!!!!111!1
        u_values[:,t] = np.array([*uold,0])
        uold = LaxWen(uold, amu)
    # creating the grids and plot
    [T, X] = np.meshgrid(tt,xx)
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.plot_wireframe(X,T,u_values)

    #ax.set_xlabel('t')
    #ax.set_ylabel('x')
    #ax.set_zlabel('u(x,t)')

    #plt.title('amu = 0.9')

    plt.plot(tt,rms_norms)
    plt.ylabel('x')
    plt.xlabel('t')
    plt.title('Plot of the norm of the solution vs time. amu = 0.9')
    plt.show()

def g(x):
    return np.exp(-100*(x-0.5)**2)

def RMS(vector): #root mean square norm
    norm = 0
    for a in range(len(vector)):
        norm = norm + vector[a]**2
    norm  = norm/(len(vector)+1)

    return norm**0.5

#advectionSolver(5, g, -1, 30,200)

#advectionSolver(5,g, 2, 20, 200) # För amu = 1 kan man ta a = 2, N = 20, och M =200 ( och t=5). Här syns konstant amplitud
advectionSolver(5,g,1.2,30,200) # För amu = 0.9. Här syns dampening
