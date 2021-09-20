#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from solvers import generate_trajectories, sample_trajectories, simulate_ODE, generate_flow
from math import comb
import random
from main import *
random.seed(a=0)
            
# =============================================================================
# Simulate dynamical system and add noise
# =============================================================================
# x0 = [0.1,0.1] 
# par = {'mu': 0.1}
# fun = 'vanderpol'

par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
fun = 'lorenz'

# par['sigma']*((par['sigma']+par['beta']+3)/(par['sigma']-par['beta']-1))


#simulate system
x0 = [-8.0, 7.0, 27.0]
t = np.linspace(0, 100, 1000)
mu, sigma = 0, 1 # mean and standard deviation
X = simulate_ODE(fun, t, x0, par, noise=True, mu=mu, sigma=sigma)

n=50
t_sample, X = sample_trajectories(X, n, T=10, t0=0.1, stack=True, seed=0)
# X0_range = [[-20,20],[-20,20],[-20,20]]
# X = generate_trajectories(fun, n, t, X0_range, P=par, seed=0, noise=False, stack=True, mu=mu, sigma=sigma)

# =============================================================================
# Obtain scalar time series by random projections (rotating the global 
# coordinate system to random angles and then taking the first coordinate)
# =============================================================================
n_obs = 50

x = []
for i in range(n_obs):
    x_tmp = random_projection(X, seed=i)
    x.append(x_tmp)

# =============================================================================
# delay embed each time series and standardize
# =============================================================================
tau = -1
dim = 3

X_nodelay, X_delay = [], []
for i in range(n_obs):
    X_tmp = delay_embed(x[i],dim,tau)
    X_tmp = standardize_data(X_tmp)
    
    #separate coordinate without delay and with delay
    X_nodelay += [X_tmp[:,0]]
    X_delay += list(X_tmp[:,1:].T)
    
#compute the number of embedding combinations where one coordinate without 
#delay is paired with two coordinates with delay
# m = comb(n_obs*dim,dim) - comb(n_obs*(dim-1),dim)
# m = int(np.sqrt(m)) #use only sqrt(m) of these embeddings
m=20

#pair one coordinate without delay with 2 delayed coordinates
X_m = []
for i in range(m):
    X_tmp = random.sample(X_nodelay, 1) + random.sample(X_delay, dim-1)
    X_m.append(np.vstack(X_tmp).T)
    

t = [0,1]
T=3

t_nn = [t]
for i in range(m):
    _, nn = find_nn(X_m[i][t], X_m[i])
    t_nn.append(list(np.squeeze(nn)))
t_nn = np.array(t_nn)
    
ts_nn, tt_nn = valid_flows(t_sample, t_nn, T=T)

#need to compute geodesic distances on the same attractor for consistency?
dst = all_geodesic_dist(X, ts_nn, tt_nn, interp=False)

kappa = curvature(ts_nn, dst)
volume_simplex(X,ts_nn)

#try with divergence of simple volumes too relative to the geodesic distance
    
# #plot attractor (because this one looks good)
# X_tmp = X_m[13]
# ax = plot_trajectories(X_tmp, color=None, style='o', lw=1, ms=2)

# #plot trajectory of neighbours
# X_neigh = generate_flow(X_tmp, t_nn, T=T, stack=False)
# ax = plot_trajectories(X_neigh, ax=ax, color='C1', style='-', lw=1, ms=50)

# #plot trajectory between two points
# X_pt = generate_flow(X_tmp, [t], T=T, stack=False)
# ax = plot_trajectories(X_pt, ax=ax, color='C3', style='-', lw=1, ms=50)