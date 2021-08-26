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

#plot
fig = plt.figure()
ax1 = fig.gca(projection="3d")
ax1.plot(X[:, 0], X[:, 1], X[:, 2])

#delay embed time series
tau = -1
dim = 3
Yx = delay_embed(X[:, 0],dim,tau)

#normalize attractor
Yx_norm = normalize_data(Yx)

#plot
fig = plt.figure()
ax2 = fig.gca(projection="3d")
ax2.plot(Yx_norm[:, 0], Yx_norm[:, 1], Yx_norm[:, 2])

#find nearest neighbor
nb=find_nn([Yx[0]],Yx,nn=2)
