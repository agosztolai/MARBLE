import matplotlib.pyplot as plt
import numpy as np
from solvers import simulate_ODE
from main import *
import plotting as plot

"""
1. Random projection to scalar time series 
2. Delay embedding
"""

#Lorenz system
x0 = [-8.0, 7.0, 27.0] 
par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
fun = 'lorenz'

#simulate Lorenz system
t = np.linspace(0, 20, 1000)
X = simulate_ODE(fun, t, x0, par)

plot.trajectories(X, color=None, style='o', lw=1, ms=1)

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

#plot embedding
plot.trajectories(Yx_norm, color=None, style='o', lw=1, ms=1)
