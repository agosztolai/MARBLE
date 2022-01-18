#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
# from math import comb
import random
from GeoDySys import time_series, curvature, plotting, solvers
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
tau = -1
dim = 3
X = time_series.delay_embed(X[:,0],dim,tau)
t_sample = t_sample[:-dim]
 

"""3. Compute curvature of trajectories starting at every point"""

times = [3] #time horizon
# n=200
# t = random.sample(list(np.arange(X.shape[0])), n)

_, nn = time_series.find_nn(X[t_sample], X, nn=10, nmax=10)
t_nn = np.hstack([np.array(t_sample)[:,None],np.array(nn)])

kappas = []
for T in times:
    #checks if the trajectory ends before time horizon T
    ts, tt = time_series.valid_flows(t_sample, t_nn, T=T)
    
    #computes geodesic distances on attractor X
    dst = curvature.all_geodesic_dist(X, ts, tt, interp=False)
    
    #computes curvatures of the geodesics
    kappa = curvature.curvature_geodesic(dst)
    kappas.append(kappa)

kappas = np.array(kappas)
# plot.plot_curvatures(times,kappas,ylog=True)

kappa = np.clip(kappas[0], -0.1, 0.1)

"""Plotting"""
# ax = plot.trajectories(X, color=None, style='o', lw=1, ms=1)
# flows_n = time_series.generate_flow(X, ts[55,1:], T=T)
# plot.trajectories(flows_n, ax=ax, color='C1', style='-', lw=1, ms=4)
# flow = time_series.generate_flow(X, ts[55,0], T=T)
# plot.trajectories(flow, ax=ax, color='C3', style='-', lw=1, ms=4)

flows = time_series.generate_flow(X, ts[:,0], T=T)
ax = plotting.trajectories(flows, color=kappa, style='-', lw=0.5, ms=6)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
plt.savefig('../results/manifold.svg')

plotting.time_series(t,X[:,0], color=kappa, style='-', lw=2)