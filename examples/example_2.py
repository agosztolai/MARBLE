#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
# from math import comb
import random
from GeoDySys import time_series, curvature, plotting, solvers, discretisation, op_calculations
random.seed(a=0)
            
"""1. Simulate system"""
# x0 = [0.1,0.1] 
# par = {'mu': 0.1}
# fun = 'vanderpol'

par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
fun = 'lorenz'

# par['sigma']*((par['sigma']+par['beta']+3)/(par['sigma']-par['beta']-1))

n=1000
x0 = [-8.0, 7.0, 27.0]
t = np.linspace(0, 20, n)
mu, sigma = 0, 1 # mean and standard deviation
X = solvers.simulate_ODE(fun, t, x0, par, noise=False, mu=mu, sigma=sigma)
t_sample = np.arange(X.shape[0]) 


"""2. Random project and then delay embed"""
x = time_series.random_projection(X, seed=0)
dim = 3
X = time_series.delay_embed(X[:,0],dim,tau=-1)
t_sample = t_sample[:-dim]


"""3. Discretisation and Markov transition operator"""
centers, sizes, labels = discretisation.kmeans_part(X, 40)
# centers, sizes, labels = discretisation.maxent_part(X, dim, .1)

P = op_calculations.get_transition_matrix(t_sample,labels,T=5)
 

"""4. Curvature of trajectories starting at the centerpoint of clusters"""
times = [3] #time horizon
# n=200
# t = random.sample(list(np.arange(X.shape[0])), n)

K = curvature.get_curvature_matrix(X,t_sample,labels,T=5,Tmax=20)

# _, nn = time_series.find_nn(X[t_sample], X, nn=10, nmax=10)
# t_nn = np.hstack([np.array(t_sample)[:,None],np.array(nn)])

# kappas = []
# for T in times:
    
#     #checks if the trajectory ends before time horizon T
#     ts, tt = time_series.valid_flows(t_sample, t_nn.flatten(), T=T)
#     ts = ts.reshape(t_nn.shape)
#     tt = tt.reshape(t_nn.shape)
    
#     #computes geodesic distances on attractor X
#     dst = curvature.all_geodesic_dist(X, ts, tt, interp=False)
    
#     #computes curvatures of the geodesics
#     kappa = curvature.curvature_geodesic(dst)
#     kappas.append(kappa)

kappas = np.array(kappas)

kappa = np.clip(kappas[0], -0.1, 0.1)


"""Plotting"""
#discretisation
ax = plotting.plot_discretisation(centers, sizes)

#transition matrix
plt.figure()
plt.imshow(P.todense())

#curvature across time horizons
# plot.plot_curvatures(times,kappas,ylog=True)

#plotting some trajectories and their neighbours
ax = plotting.trajectories(X, color=None, style='o', lw=1, ms=1)
flows_n = time_series.generate_flow(X, ts[55,1:], T=T)
plotting.trajectories(flows_n, ax=ax, color='C1', style='-', lw=1, ms=4)
flow = time_series.generate_flow(X, ts[55,0], T=T)
plotting.trajectories(flow, ax=ax, color='C3', style='-', lw=1, ms=4)

#plotting all curvatures
flows = time_series.generate_flow(X, ts[:,0], T=T)
ax = plotting.trajectories(flows, color=kappa, style='-', lw=0.5, ms=6)

# plt.savefig('../results/manifold.svg')

#plot trajectory with curvature values
ax = plotting.time_series(t,X[:,0], color=kappa, style='-', lw=2)
ax.set_xlim([0,20])