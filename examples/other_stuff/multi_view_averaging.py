#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from solvers import generate_trajectories, sample_trajectories, simulate_ODE
from math import comb
import random
from main import *
import plotting as plot
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
t = np.linspace(0, 20, 1000)
mu, sigma = 0, 1 # mean and standard deviation
X = simulate_ODE(fun, t, x0, par, noise=False, mu=mu, sigma=sigma)

# X = X[20:]
# plot.trajectories(X, color=None, style='-', lw=1, ms=1)

#simulate having short trajectories by sampling from the manifold
# n=50
# t_sample, X = sample_trajectories(X, n, T=20, t0=0.1, seed=0)
# X0_range = [[-20,20],[-20,20],[-20,20]]
# X = generate_trajectories(fun, n, t, X0_range, P=par, seed=0, noise=False, stack=True, mu=mu, sigma=sigma)

# X = stack(X)

# =============================================================================
# Obtain scalar time series by random projections (rotating the global 
# coordinate system to random angles and then taking the first coordinate)
# =============================================================================
# n_obs = 1

# x = []
# for i in range(n_obs):
#     x_tmp = random_projection(X, seed=i)
#     x.append(x_tmp)

# =============================================================================
# delay embed each time series and standardize
# =============================================================================
tau = -1
dim = 3

# X = delay_embed(X[:,0],dim,tau)

# X_nodelay, X_delay = [], []
# for i in range(n_obs):
#     X_tmp = delay_embed(x[i],dim,tau)
#     X_tmp = standardize_data(X_tmp)
    
#     #separate coordinate without delay and with delay
#     X_nodelay += [X_tmp[:,0]]
#     X_delay += list(X_tmp[:,1:].T)
    
    
n=200
T=10
t = random.sample(list(np.arange(X.shape[0])), n)


_, nn = find_nn(X[t], X, nn=10, nmax=10)
t_nn = np.hstack([np.array(t)[:,None],np.array(nn)])

#compute the number of embedding combinations where one coordinate without 
#delay is paired with two coordinates with delay
# m = comb(n_obs*dim,dim) - comb(n_obs*(dim-1),dim)
# m = int(np.sqrt(m)) #use only sqrt(m) of these embeddings
# m=20

#pair one coordinate without delay with 2 delayed coordinates
# X_m = []
# for i in range(m):
#     X_tmp = random.sample(X_nodelay, 1) + random.sample(X_delay, dim-1)
#     X_m.append(np.vstack(X_tmp).T)

#take one nearest neighbor in each attractor
# t_nn = [t]
# for i in range(m):
    #take one nearest neighbor in each attractor
    # _, nn = find_nn(X_m[i][t], X_m[i])
    # t_nn.append(list(np.squeeze(nn)))
# t_nn = np.array(t_nn)

t_sample = np.arange(X.shape[0])
ts, tt = valid_flows(t_sample, t_nn, T=T)

#need to compute geodesic distances on the same attractor for consistency?
dst = all_geodesic_dist(X, ts, tt, interp=False)
kappa = curvature_geodesic(dst)
kappa = np.clip(kappa, -0.1, 0.1)
# kappa = curvature_ball(X, ts_nn, tt_nn)

# =============================================================================
# some plots
# =============================================================================
ax = plot.trajectories(X, color=None, style='o', lw=1, ms=1)
flows_n = generate_flow(X, ts[55,1:], T=T)
plot.trajectories(flows_n, ax=ax, color='C1', style='-', lw=1, ms=4)
flow = generate_flow(X, ts[55,0], T=T)
plot.trajectories(flow, ax=ax, color='C3', style='-', lw=1, ms=4)

flows = generate_flow(X, ts[:,0], T=T)
ax = plot.trajectories(flows, color=kappa, style='-', lw=0.5, ms=6)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
plt.savefig('manifold.svg')



