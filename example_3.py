#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from solvers import simulate_ODE
from math import comb
from random import sample
from main import *

n_obs = 50
tau = -1
dim = 3
L = 1000
T = 10
mu, sigma = 0, 1 # mean and standard deviation

#Lorenz system
x0 = [-8.0, 7.0] 
par = {'mu': 1, 'omega': 1, 'b': 1}
fun = 'sup_hopf'

#simulate Lorenz system
t = np.linspace(0, T, L)
X = simulate_ODE(fun, t, x0, par)
X += np.random.normal(mu, sigma, size = (L,2))

fig = plt.figure()
ax0 = plt.axes()
ax0.plot(t, X)

# #Obtain scalar time series by random projections (rotating the global 
# #coordinate system to random angles and then taking the first coordinate)
# x = []
# for i in range(n_obs):
#     x_tmp = random_projection(X, seed=i)
#     x.append(x_tmp)

# #delay embed each time series and standardize
# X, X_nodelay, X_delay = [], [], []
# for i in range(n_obs):
#     X_tmp = delay_embed(x[i],dim,tau)
#     X_tmp = standardize_data(X_tmp)
    
#     X.append(X_tmp)
    
#     #separate coordinate without delay and with delay
#     X_nodelay += [X_tmp[:,0]]
#     X_delay += list(X_tmp[:,1:].T)
    
# #compute the number of embedding combinations where one coordinate without 
# #delay is paired with two coordinates with delay
# m = comb(n_obs*dim,dim) - comb(n_obs*(dim-1),dim)
# m = int(np.sqrt(m)) #use only sqrt(m) of these embeddings

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


