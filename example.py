import matplotlib.pyplot as plt
import numpy as np
from solvers import simulate_ODE
from main import *

#Lorenz system
x0 = [-8.0, 7.0, 27.0] 
par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
fun = 'lorenz'

#simulate Lorenz system
t = np.linspace(0, 20, 1000)
X = simulate_ODE(fun, t, x0, par)

fig = plt.figure()
ax1 = plt.axes(projection="3d")
ax1.plot(X[:, 0], X[:, 1], X[:, 2])

#project to random scalar timeseries
x = random_projection(X)
fig = plt.figure()
ax2 = plt.axes()
ax2.plot(t, x)

#delay embed time series
tau = -1
dim = 3
Yx = delay_embed(x,dim,tau)

#normalize attractor
Yx_norm = standardize_data(Yx)

fig = plt.figure()
ax3 = plt.axes(projection="3d")
ax3.plot(Yx_norm[:, 0], Yx_norm[:, 1], Yx_norm[:, 2])

#find nearest neighbor
nb=find_nn([Yx[0]],Yx,nn=2)

#find geodesic distance between two points 
dist = geodesic_dist(0, 10, Yx, interp=False)
print(dist)
