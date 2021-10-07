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
            
"""
1. Simulate system
2. Sample trajectories from attractor
3. Compute curvature around these trajectories
"""

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
t_sample = np.arange(X.shape[0]) 
 
n=200
# T=10
times = np.arange(1,20)
t = random.sample(list(np.arange(X.shape[0])), n)
t = np.array(t)

_, nn = find_nn(X[t], X, nn=10, nmax=10)
t_nn = np.hstack([t[:,None],np.array(nn)])

kappas = []
for T in times:
    ts, tt = valid_flows(t_sample, t_nn, T=T)

    #need to compute geodesic distances on the same attractor for consistency?
    dst = all_geodesic_dist(X, ts, tt, interp=False)
    kappa = curvature_geodesic(dst)
    kappas.append(kappa)
    # kappa = curvature_ball(X, ts_nn, tt_nn)

kappas = np.array(kappas)
plot.plot_curvatures(times,kappas,ylog=True)

T = 5
kappa = np.clip(kappas[T], -0.1, 0.1)

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
# plt.savefig('manifold.svg')


#form nn graph
import networkx as nx
X_nodes = X[t,:]
t = np.arange(n)
t = np.delete(t,t[np.isnan(kappa)])
X_nodes = X_nodes[t]
_, nn = find_nn(X_nodes, X_nodes, nn=5, nmax=10)

kappa_diff = kappa[:,None] - kappa[:,None].T
G = nx.Graph()
G.add_nodes_from([i for i in range(n)])
for i,nn_ in enumerate(nn):
    x = [i] * 5
    G.add_weighted_edges_from(zip(x,nn_,kappa_diff[i,nn_]))
    
node_colors = kappa

plot.plot_graph(G,node_colors=node_colors)
