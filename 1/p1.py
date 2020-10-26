# This example describe how to integrate ODEs with scipy.integrate module, and how
# to use the matplotlib module to plot trajectories, direction fields and other
# useful information.
# 
# == Presentation of the Lokta-Volterra Model ==
# 
# We will have a look at the Lokta-Volterra model, also known as the
# predator-prey equations, which are a pair of first order, non-linear, differential
# equations frequently used to describe the dynamics of biological systems in
# which two species interact, one a predator and one its prey. They were proposed
# independently by Alfred J. Lotka in 1925 and Vito Volterra in 1926:
# du/dt =  a*u -   b*u*v
# dv/dt = -c*v + d*u*v 
# 
# with the following notations:
# 
# *  u: number of preys (for example, rabbits)
# 
# *  v: number of predators (for example, foxes)  
#   
# * a, b, c, d are constant parameters defining the behavior of the population:    
# 
#   + a is the natural growing rate of rabbits, when there's no fox
# 
#   + b is the natural dying rate of rabbits, due to predation
# 
#   + c is the natural dying rate of fox, when there's no rabbit
# 
#   + d is the factor describing how many caught rabbits let create a new fox
# 
# We will use X=[u, v] to describe the state of both populations.
# 
# Definition of the equations:
# 
from numpy import *
import pylab as p
from scipy import integrate


def dX_dt(t, x):
    """ Return the growth rate of fox and rabbit populations. """
    rabbits, foxes = x
    drabbitdt = a * rabbits - b * rabbits * foxes
    dfoxesdt = -c * foxes + d  * rabbits * foxes
    return array([drabbitdt, dfoxesdt])

# Definition of parameters 
a = 0.1
b = 0.02
c = 0.3
d = 0.01

# RungeKutta args
h = 0.05
t1 = 0
tf = 200


# === Population equilibrium ===
# 
# Before using !SciPy to integrate this system, we will have a closer look on 
# position equilibrium. Equilibrium occurs when the growth rate is equal to 0.
# This gives two fixed points:
# 
X_f0 = array([ 0.,  0.])
X_f1 = array([ c/(d), a/b])
all(dX_dt(None,X_f0) == zeros(2) ) and all(dX_dt(None,X_f1) == zeros(2)) # => True 

# == Plotting direction fields and trajectories in the phase plane ==
# 
# We will plot some trajectories in a phase plane for different starting
# points between X_f0 and X_f1.
# 
# We will use matplotlib's colormap to define colors for the trajectories.
# These colormaps are very useful to make nice plots.
# Have a look at [http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps ShowColormaps] if you want more information.
# 
#-------------------------------------------------------
#Ejercicio1.

## Parámetros de integración
h  = 0.1                          ## Paso temporal 
t0 = 0                            ## Tiempo inicial
tf = 100                           ## Tiempo final
t_span = (t0, tf)                 ## Intervalo temporal
## Instantes donde evaluar la solución.
t_eval = arange(t0, tf, h) 

condiciones_iniciales = [

  (2,2),
]
f2 = p.figure()

#-------------------------------------------------------
# plot trajectories
soluciones = []
for i, y0 in enumerate(condiciones_iniciales): 
    # X0 = y0 * X_f1                               # starting point
    sol = integrate.solve_ivp(dX_dt, t_span=t_span, y0=y0, t_eval=t_eval, method='RK45')         # we don't need infodict here
    soluciones.append(sol)
#-------------------------------------------------------
# define a grid and compute direction at each point
ymax = p.ylim(ymin=0)[1]                        # get axis limits
xmax = p.xlim(xmin=0)[1] 
#ymax = 15
#xmax = 50
nb_points   = 20                      

x = linspace(0, xmax, nb_points)
y = linspace(0, ymax, nb_points)

X1 , Y1  = meshgrid(x, y)                       # create a grid
DX1, DY1 = dX_dt(None,[X1, Y1])                      # compute growth rate on the gridt
M = (hypot(DX1, DY1))                           # Norm of the growth rate 
M[ M == 0] = 1.                                 # Avoid zero division errors 
DX1 /= M                                        # Normalize each arrows
DY1 /= M                                  

#-------------------------------------------------------
# Drow direction fields, using matplotlib 's quiver function
# I choose to plot normalized arrows and to use colors to give information on
# the growth speed
fig, ax = p.subplots()
#ax.title('Diagrama de flujo')
Q = ax.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=p.cm.jet)
ax.plot(X_f0[0],X_f0[1],'o', fillstyle='none')
ax.plot(X_f1[0],X_f1[1],'o', fillstyle='none')
ax.set_ylabel('Conejos')
ax.set_xlabel('Zorros')
ax.legend()
ax.grid()

ax.set_xlim(0, 80)
ax.set_ylim(0, 30)
for i, y0 in enumerate(condiciones_iniciales):
  sol = soluciones[i]
  ax.plot(sol.y[0], sol.y[1])
p.savefig('rabbits_and_foxes_2.png')
# 
# 
# We can see on this graph that an intervention on fox or rabbit populations can
# have non intuitive effects. If, in order to decrease the number of rabbits,
# we introduce foxes, this can lead to an increase of rabbits in the long run,
# if that intervention happens at a bad moment.
# 
# 
#-------------------------------------------------------
#Ejercicio2
# RungeKutta args
#Resuelvo el ejercicio 2. Encuentro solucion numerica. Utilizo RungeKutta.
h = 0.05
t0 = 0
tf = 200
t_span = (t0, tf)

## Instantes donde evaluar la solución.
t_eval = arange(t0, tf, h) 

# Initial conditions
x0 = 40
y0 = 9
initial_conditions = array([x0, y0])


## Integramos la solución
for i, y0 in enumerate(initial_conditions):
    solution = integrate.solve_ivp(
        dX_dt, 
        t_span=t_span,
        y0=initial_conditions, 
        t_eval=t_eval,
        method='RK45' ## Resolvemos utilizando el método de Runge-Kutta de orden 4

    )

p.figure("C(t) vs Z(t)", figsize=(8,5))
p.title("Evolucion Temporal")
p.plot(t_eval, solution.y[0], label='Conejos')
p.plot(t_eval, solution.y[1], label='Zorros')
p.xlabel('Tiempo')
p.ylabel('Población')
p.legend()


#-------------------------------------------------------
#Ejercicio3
# Mismo que uno pero con valores concreto.
p.figure("Diagrama de Flujo para 40 conejos y 9 zorros", figsize=(8,5))
p.title("Diagrama de Flujo para 40 conejos y 9 zorros")
p.plot(X_f1[0],X_f1[1],'o')
p.legend()
p.xlabel('Tiempo')
p.ylabel('Población')
p.legend()
p.xlim(0, xmax)
p.ylim(0, ymax)
p.grid()

p.show()

# Q = p.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=p.cm.jet)


#sol = solve_ivp(dX_dt, t_span=t_span, y0=y0, t_eval=t_eval)
# Esto es la definicion de Runge Kutta, no hace falta. Lo resuelvo como arriba.
# Finds value of y for a given x using step size h 
# and initial value y0 at x0. 
# def rungeKutta(x0, y0, x, h): 
#     # Count number of iterations using step size or 
#     # step height h 
#     n = (int)((x - x0)/h)  
#     # Iterate for number of iterations 
#     y = y0 
#     for i in range(1, n + 1): 
#         "Apply Runge Kutta Formulas to find next value of y"
#         k1 = h * df_dt(x0, y) 
#         k2 = h * df_dt(x0 + 0.5 * h, y + 0.5 * k1) 
#         k3 = h * df_dt(x0 + 0.5 * h, y + 0.5 * k2) 
#         k4 = h * df_dt(x0 + h, y + k3) 

#         # Update next value of y 
#         y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 

#         # Update next value of x 
#         x0 = x0 + h 
#         return y





# # 
# # === Stability of the fixed points ===
# # Near theses two points, the system can be linearized:
# # dX_dt = A_f*X where A is the Jacobian matrix evaluated at the corresponding point.
# # We have to define the Jacobian matrix:
# # 
# def d2X_dt2(X, t=0):
#     """ Return the Jacobian matrix evaluated in X. """
#     return array([[a -b*X[1],   -b*X[0]     ],
#                   [d*X[1] ,   -c +d*X[0]] ])  
# # 
# # So, near X_f0, which represents the extinction of both species, we have:
# # A_f0 = d2X_dt2(X_f0)                    # >>> array([[ 1. , -0. ],
# #                                         #            [ 0. , -1.5]])
# # 
# # Near X_f0, the number of rabbits increase and the population of foxes decrease.
# # The origin is a [http://en.wikipedia.org/wiki/Saddle_point saddle point].
# # 
# # Near X_f1, we have:
# A_f1 = d2X_dt2(X_f1)                    # >>> array([[ 0.  , -2.  ],
#                                         #            [ 0.75,  0.  ]])

# # whose eigenvalues are +/- sqrt(c*a).j:
# lambda1, lambda2 = linalg.eigvals(A_f1) # >>> (1.22474j, -1.22474j)

# # They are imaginary number, so the fox and rabbit populations are periodic and
# # their period is given by:
# T_f1 = 2*pi/abs(lambda1)                # >>> 5.130199
# #         
# # == Integrating the ODE using scipy.integate ==
# # 
# # Now we will use the scipy.integrate module to integrate the ODEs.
# # This module offers a method named odeint, very easy to use to integrate ODEs:
# # 

# t = linspace(0, 100,  800)              # time
# X0 = array([40, 9])                     # initials conditions: 40 rabbits and 9 foxes  

# X, infodict = integrate.odeint(dX_dt, X0, t, full_output=True)
# infodict['message']                     # >>> 'Integration successful.'
# # 
# # `infodict` is optional, and you can omit the `full_output` argument if you don't want it.
# # Type "info(odeint)" if you want more information about odeint inputs and outputs.
# # 
# # We can now use Matplotlib to plot the evolution of both populations:
# # 
# rabbits, foxes = X.T

# f1 = p.figure()
# p.plot(t, rabbits, 'r-', label='Rabbits')
# p.plot(t, foxes  , 'b-', label='Foxes')
# p.grid()
# p.xlabel('Time')
# p.ylabel('Population')
# p.title('Evolucion de las poblaciones de Conejos y Zorros')
# f1.savefig('rabbits_and_foxes_1.png')
# 
# 
# The populations are indeed periodic, and their period is near to the T_f1 we calculated.
# 

# == Plotting contours ==
# 
# We can verify that the function IF defined below remains constant along a trajectory:
# 
# def IF(X):
#     u, v = X
#     return u**(c/a) * v * exp( -(b/a)*(d*u+v) )

# # We will verify that IF remains constant for different trajectories
# for v in values: 
#     X0 = v * X_f1                               # starting point
#     X = integrate.odeint( dX_dt, X0, t)         
#     I = IF(X.T)                                 # compute IF along the trajectory
#     I_mean = I.mean()
#     delta = 100 * (I.max()-I.min())/I_mean
    #print 'X0=(%2.f,%2.f) => I ~ %.1f |delta = %.3G %%' % (X0[0], X0[1], I_mean, delta)

# >>> X0=( 6, 3) => I ~ 20.8 |delta = 6.19E-05 %
#     X0=( 9, 4) => I ~ 39.4 |delta = 2.67E-05 %
#     X0=(12, 6) => I ~ 55.7 |delta = 1.82E-05 %
#     X0=(15, 8) => I ~ 66.8 |delta = 1.12E-05 %
#     X0=(18, 9) => I ~ 72.4 |delta = 4.68E-06 %
# 
# Potting iso-contours of IF can be a good representation of trajectories,
# without having to integrate the ODE
# 
#-------------------------------------------------------
# plot iso contours
# nb_points = 80                              # grid size 

# x = linspace(0, xmax, nb_points)    
# y = linspace(0, ymax, nb_points)

# X2 , Y2  = meshgrid(x, y)                   # create the grid
# Z2 = IF([X2, Y2])                           # compute IF on each point

# f3 = p.figure()
# CS = p.contourf(X2, Y2, Z2, cmap=p.cm.Purples_r, alpha=0.5)
# CS2 = p.contour(X2, Y2, Z2, colors='black', linewidths=2. )
# p.clabel(CS2, inline=1, fontsize=16, fmt='%.f')
# p.grid()
# p.xlabel('Rabbits')
# p.ylabel('Foxes')
# p.ylim(1, ymax)
# p.xlim(1, xmax)
# p.title('IF contours')
# f3.savefig('rabbits_and_foxes_3.png')
# p.show()
# 
# 
# # vim: set et sts=4 sw=4:
