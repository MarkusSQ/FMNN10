# PART 4
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg as lg
import scipy.sparse as sp
import scipy.sparse.linalg
from Project3.Part1 import TRstep, g

def h(x):
    return 3*np.exp(-100*(x-0.5)**2)

# PART 4
def LW(uold, dt, dx):
  N = len(uold)
  unew = np.zeros(N)
  
  for i in range(N):
    unew[i] = uold[i] - dt/(2*dx)*uold[i]*(uold[(i+1)%N]-uold[(i-1)%N]) + (dt**2)*uold[i]*((uold[(i+1)%N]-uold[(i-1)%N])/(2*dx))**2 + (dt**2/2)*(uold[i]**2)*(uold[(i+1)%N]-2*uold[i]+uold[(i-1)%N])/(dx**2)
  #unew[0] = 0
  #unew[-1] = unew[0]
  return unew

def burgerEater(N, M, g, d, tend):
  dx = 1/N
  dt = tend/M

  xx = np.linspace(0,1, num = N+1)
  tt = np.linspace(0, tend, num = M+1)

  row = np.zeros(N)
  row[0] = -2/(dx**2)
  row[1] = 1/(dx**2)
  t_matrix = lg.toeplitz(row, row)
  t_matrix[N-1][0] = 1/(dx**2) # super
  t_matrix[0][N-1] = 1/(dx**2) # sub

  I = np.identity(N) #ändrade från N

  uold = g(np.linspace(0,1-dx, num=N)) # tog bort 1-dx och bytte N mot N+1
  
  u_values = np.zeros((N+1,M+1))

  for t in range(M):
    #u_values[:,t] = uold
    u_values[:,t] = np.array([*uold, 0])
    uold = np.linalg.solve(I-d*dt*0.5*t_matrix, LW(uold,dt,dx) + d*dt*0.5*np.matmul(t_matrix,uold))
  
  [T, X] = np.meshgrid(tt,xx)
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot_wireframe(X,T,u_values)

  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_zlabel('$u$')

  plt.title('N=300, M=1000, d=0.1, g(x) = 3*np.exp(-100*(x-0.5)**2)')

  plt.show()

burgerEater(300, 1000, h, 0.01, 1)