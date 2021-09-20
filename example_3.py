#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from solvers import generate_trajectories, sample_trajectories, simulate_ODE, generate_flow
from math import comb
from random import sample
from main import *

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
X = simulate_ODE(fun, t, x0, par, noise=False, mu=mu, sigma=sigma)

n=50
X = sample_trajectories(X, n, T=10, t0=0.1, stack=True, seed=0)
# X0_range = [[-20,20],[-20,20],[-20,20]]
# X = generate_trajectories(fun, n, t, X0_range, P=par, seed=0, noise=False, stack=True, mu=mu, sigma=sigma)

plot_trajectories(X, color=None, style='o', lw=1, ms=2)

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

X, X_nodelay, X_delay = [], [], []
for i in range(n_obs):
    X_tmp = delay_embed(x[i],dim,tau)
    X_tmp = standardize_data(X_tmp)
    
    X.append(X_tmp)
    
    #separate coordinate without delay and with delay
    X_nodelay += [X_tmp[:,0]]
    X_delay += list(X_tmp[:,1:].T)
    
#compute the number of embedding combinations where one coordinate without 
#delay is paired with two coordinates with delay
m = comb(n_obs*dim,dim) - comb(n_obs*(dim-1),dim)
m = int(np.sqrt(m)) #use only sqrt(m) of these embeddings

# dist_nn, ind_nn = find_nn(X[0,None], X, nn=5, n_jobs=-1)
# X_neigh = generate_flow(X, ind_nn, T=5, stack=False)
# ax = plot_trajectories(X_neigh, ax=ax, color='C1', style='-', lw=1, ms=10)

# # fig = plt.figure()
# # ax2 = plt.axes(projection="3d")
# #pair one coordinate without delay with 2 delayed coordinates
# X_m = []
# for i in range(m):
#     X_tmp = sample(X_nodelay, 1) + sample(X_delay, dim-1)
#     X_m.append(np.vstack(X_tmp).T)
#     # ax2.plot(X_m[i][:, 0], X_m[i][:, 1], X_m[i][:, 2])
    
# obs_i = 5
# t = np.random.randint(L-(dim-1)*abs(tau))

# fig = plt.figure()
# ax3 = plt.axes(projection="3d")
# ax3.plot(X[obs_i][:, 0], X[obs_i][:, 1], X[obs_i][:, 2])

# x_pred = 0
# for i in range(m):
#     _, nn_t = find_nn(X_m[i][t][None], X_m[i])

#     nn_t = np.squeeze(nn_t)[1:]
#     x_pred += X[obs_i][nn_t,0] #perhaps weight neighbours?
#     ax3.scatter(X[obs_i][nn_t, 0], X[obs_i][nn_t, 1], X[obs_i][nn_t, 2],c='g')
  
# print(X[obs_i][t][0] - x_pred/m)

# ax3.scatter(X[obs_i][t, 0],X[obs_i][t, 1],X[obs_i][t, 2],c='r')


# #find geodesic distance between two points 
# # dist = geodesic_dist(0, 10, Yx, interp=False)
# # print(dist)


