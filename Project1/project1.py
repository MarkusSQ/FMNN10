import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lg
import scipy.integrate as integ

def f(t,y): #definierar vår diffekvation
    fval = 1*y #lambda valt till 1
    return  fval

def RK4step(f, told , uold, h): #tar ett steg enligt explicit klassisk runge kutta (uppg 1.1) 
    y1 = f(told, uold)
    y2 = f(told+h/2, uold+h*y1/2)
    y3 = f(told+h/2,uold+h*y2/2)
    y4 = f(
        told+h,uold+h*y3)
    unew = uold + (h/6)*(y1+2*y2+2*y3+y4)
    return unew

def RK34step(f, told , uold, h): #tar ett steg enligt explicit runge kutta 
    y1 = f(told, uold)
    y2 = f(told+h/2, uold+h*y1/2)
    y3 = f(told+h/2,uold+h*y2/2)
    z3 = f(told+h, uold-h*y1+2*h*y2) #ny evaluation
    y4 = f(told+h,uold+h*y3)
    
    unew = uold + (h/6)*(y1+2*y2+2*y3+y4)
    err = lg.norm((h/6)*(2*y2+z3-2*y3-y4)) #felet, euklidiska normen av l_(n+1) (l_n+1 ges av z_(n+1)-y_(n+1))
    return [unew, err]


def RKint(f, A, y0, t0, tf, N):
    uold = y0
    h = (tf-t0)/N #step size
    tgrid = np.linspace(t0,tf,num=N)
    results = []
    lerror = []
    relerror = []
    for x in range(N+1): #x are all steps we take so can be compared to t0, t1 ...
        results.append(uold)
        #lerror.append(lg.norm(uold-y0*lg.expm(A*(t0+h*x))))
        #relerror.append(lerror[x]/(lg.norm(y0*lg.expm(A*(t0+h*x)))))
        unew = RK4step(f, t0+h*x, uold, h) #t0 + h*x ger rätt tn
        uold = unew
    approx = results[-1]
    err = approx - y0*lg.expm(A*(tf))
    return [tgrid, approx, err, results, lerror, relerror]

def errVShRK(f, A, y0, t0, tf):
    hrange = 1000
    errors = []
    hs = []
    controlh = []
    for x in range(1,hrange):
        h = (tf-t0)/(x)
        temp = RKint(f,A,y0,t0,tf,x)[2] #ger approx
        errors.append(lg.norm(temp))
        hs.append(h) # Vi vill ha h inte N vilket är varför vi delar
        controlh.append(pow(h,4))
    return [errors, hs, controlh]

# def newstep(tol, err, errold, hold, k):
#     hnew = pow((tol/err), (2/(3*k)))*pow((tol/errold), ((-1)/(3*k)))*hold #Räknar ut hn från ekv. 1
#     return hnew

def newstep(tol, err, errold, hold, k):
    hnew = pow((tol/err),2/(3*k))*pow((tol/errold),-1/(3*k))*hold #hnew är den nya step size som behövs för att vi ska hålla vårt lokala fel inom området tol
    return hnew

def adaptiveRK34(f, t0, tf, y0, tol): #gjord för 1.4. Använder sig av RK34 samt justerar steglängd för att hålla det lokala felet under tol
    counter = 0
    
    h0 = (np.abs(tf-t0)*pow(tol, 0.25))/(100*(1+lg.norm(f(t0,y0)))) #vi har med t0 i f för att vi inte ska få problem
    h = h0

    t = t0
    y = y0

    tgrid = []
    yval = []

    errors = []
    errold = tol
    err = 1

    while t<tf:
        tgrid.append(t)
        yval.append(y)
        results = RK34step(f,t,y, h)
        y = results[0]
        t = h+t # Vi ser till att öka t
        hnew = newstep(tol, err, errold,h, 4) #vi ändrar vår steglängd för att bli bra
        if t+h >tf: #vi vill hamna exakt på tf. 
            hnew = tf-t
        h = hnew

        errold = err # vi ser till att öka errold
        err = results[1]
        errors.append(err)
        counter = counter +1 #Till för att räkna antal steg vi behöver för att lösa integration i 3.2
    return [tgrid, yval, errors, counter]

#uppgift 1.1
#array = errVShRK(f,np.matrix([1]), 1, 0, 10)
#plt.loglog(array[1],array[0])
#plt.loglog(array[1],array[2]) #Vi ser att vi har samma lutning mellan h och det globala felet => samma ordning
#plt.show()

#uppgift 1.4
#array = adaptiveRK34(f, 0, 10, 1, pow(10,-6))
#plt.plot(array[0], array[1])
#plt.show()

# Uppgift 2.1
# a = 3 # för att kunna kommas åt på andra ställen också
# b = 9
# c = 15
# d = 15

# def lotka(t,u):
#     dudt = np.array([a*u[0] - b*u[0]*u[1], c*u[0]*u[1] - d*u[1]])
#     return dudt


##Testa lotka: #strunta i denna
 #iv = np.array([1,1]) #ursprungsvärden
 #array21 = adaptiveRK34(lotka, 0, 5, iv, pow(10,-8))
 #plt.plot(array21[0],array21[1])
 #plt.show()


#Paulinas:
# initialValues_lotka = np.array([1,1])
# array3 = adaptiveRK34(lotka, 0, 500, initialValues_lotka, pow(10,-6))
# yValues = []
# xValues = []
# H = []
# Hcheck = []

# for x in range(len(array3[1])):
#     currentArray = array3[1][x]
#     xVal = currentArray[0]
#     yVal = currentArray[1]
#     xValues.append(xVal)
#     yValues.append(yVal)
#     H.append(c*xVal+b*yVal-d*np.log(xVal)-a*np.log(yVal))
#     Hcheck.append(np.abs(H[x]/(H[0]-1)))

#plt.plot(array3[0], xValues, label="rabbits")
#plt.plot(array3[0], yValues, label="wolves")

# plt.plot(array3[0], Hcheck) #för H
# plt.yscale('log')
# plt.show()

#plt.plot(xValues,yValues)
#plt.plot(array3[0], array3[1])
#plt.legend()
#plt.show()


#börjanp
# Task 2.1
a = 3
b = 9
c = 15
d = 15

def lotka(t, u): 
    return np.array([a*u[0] - b*u[0]*u[1] , c*u[0]*u[1] - d*u[1]])

initialValues_lotka = np.array([1,1])
x0 = initialValues_lotka[0]
y0 = initialValues_lotka[1]
array3 = adaptiveRK34(lotka, 0, 500, initialValues_lotka, pow(10,-6))
yValues = []
xValues = []
H_check = []
H_0 = c*x0 + b*y0 - d*np.log(x0) - a*np.log(y0)

#retrieving/separating the x and y values
for k in range(len(array3[1])):
    currentArray = array3[1][k]
    x = currentArray[0]
    y = currentArray[1]
    xValues.append(x)
    yValues.append(y)
    H = c*x + b*y - d*np.log(x) - a*np.log(y)
    H_check.append(np.abs(H/H_0 - 1))


#plt.plot(xValues,yValues) #fasdiagram som ger sluten cirkel

#plt.plot(array3[0], xValues, label="rabbits") # vet inte vad som hände nu
#plt.plot(array3[0], yValues, label="wolves")

#plt.legend()
#plt.show()

#plt.plot(array3[0], H_check) #för att kolla H
#plt.yscale('log')
#plt.show()
#slutp



#Task 3.1
def pol(t, u):
    m = 100
    return np.array([u[1], m*(1-pow(u[0],2))*u[1] - u[0]]) #y2 mot y2'

iv31 = np.array([2,0])
array31 = adaptiveRK34(pol, 0, 200,iv31, pow(10,-8))

y1Values = []
y2Values = []

for x in range(len(array31[1])): #bläddrar igenom arrays i arrays
    currentArray = array31[1][x]
    y1Values.append(currentArray[0])
    y2Values.append(currentArray[1])

#plt.plot(array31[0],y2Values) # ger periodiska lösn
#plt.show()


#Task 3.2
muindex = 0 #global variabel så vi slipper ändra i polmu
mu = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330]

def polmu(t, u):
    m = mu[muindex]
    return np.array([u[1], m*(1-pow(u[0],2))*u[1] - u[0]]) #y2 mot y2'



iv32 = np.array([2,0])

array32 = []
# for x in range(len(mu)): #iterera genom alla mu #3.1
#     array32.append(adaptiveRK34(polmu, 0, 0.7*mu[muindex],iv32,pow(10,-6))[3])
#     muindex = muindex +1

for x in range(len(mu)): #iterera genom alla mu #3.2. Performar mkt bättre tack vare att den inbygda funktionen arbetar implicit
     array32.append(len(integ.solve_ivp(polmu, [0, 0.7*mu[muindex]],iv32,'BDF').t))
     muindex = muindex +1

mu2 = []
for x in range(len(mu)):
    mu2.append(pow(mu[x],2))

#plt.loglog(mu, array32) #plottar mu mot N
#plt.loglog(mu, mu2) # Vi ser att N är proportionerligt mot C*mu^2. Då stiffness är proportionerligt mot N är alltså stiffness också prop mot C*mu^2

#plt.show()




