import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lg

def eulerstep(A, uold, h):
    unew = uold+h*np.matmul(A,uold)
    return unew
def eulerint(A, y0, t0, tf, N):
    uold = y0
    h = (tf-t0)/N
    tgrid = np.linspace(t0,tf,num=N)
    results = np.zeros(N)
    lerror = np.zeros(N)
    relerror = np.zeros(N)
    for x in range(N):
        results[x] = uold
        lerror[x] = lg.norm(uold-y0*lg.expm(A*(t0+h*x)))
        relerror[x] = lerror[x]/(lg.norm(y0*lg.expm(A*(t0+h*x))))
        unew = eulerstep(A, uold, h)
        uold = unew
    approx = results[N-1]
    err = approx - y0*lg.expm(A*tf)
    return [tgrid, approx, err, results, lerror, relerror]

def errVSh(A, y0, t0, tf):
    
    hrange = 15
    errors = np.zeros(hrange)
    Ns = np.zeros(hrange)
    for x in range(hrange):
        temp = eulerint(A,y0,t0,tf,pow(2,hrange-x))[2]
        errors[x] = lg.norm(temp)
        Ns[x] = (tf-t0)/pow(2,hrange-x) # Vi vill ha h inte N vilket är varför vi delar
    return [errors, Ns]

#Uppgift 1.1-1.2
#array1 = eulerint(np.matrix([-1]), np.array([1]), 0, 1, 100) 
#plt.plot(array1[0], array1[3])
#values = np.exp(3*array1[0])
#plt.plot(array1[0], values)


#uppgift 1.3
#for lamda in range(-3,3):
    
array = errVSh(np.matrix([-1]),np.matrix([1]),0,10)
plt.loglog(array[1],array[0], label = -1)

#uppgift 1.4
#for lamda in range(-10,0):
#    array = eulerint(np.matrix([lamda]),np.matrix([1]),0,1, 100)
#    plt.plot(array[0],array[4], label = lamda)
#    plt.yscale("log")
#for lamda in range(0,10):
#    array = eulerint(np.matrix([lamda]),np.matrix([1]),0,1, 100)
#    plt.plot(array[0],array[5], label = lamda)
#    plt.yscale("log")

#r
#for lamda in range(0,10):
#    A = np.matrix([[-1,10],[0,3]])
#    y0 = np.array([[1],[1]])
#    array = eulerint(A,y0,0,10, 100)
#    plt.plot(array[0],array[3], label = lamda)
#    plt.yscale("log")

#A = np.matrix([[-1,10],[0,-3]])
#y0 = np.array([[1],[1]])    
#array = errVSh(A,y0,0,10)
#plt.loglog(array[1],array[0])

#A = np.matrix([[-1,10],[0,-3]])
#y0 = np.array([[1],[1]]) 
#array = eulerint(A,y0,0,10, 100)
#plt.plot(array[0],array[4])
#plt.yscale("log")

#print(lg.eig(A))
plt.legend()
plt.show()


        