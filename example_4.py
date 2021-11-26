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
times = np.arange(1,50)
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

T = 3
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

ax2 = plot.trajectories(flows, color=kappa, style='-', lw=0.5, ms=6)

#form nn graph
numn=10

import networkx as nx
X_nodes = X[t]
t = np.arange(n)
t = np.delete(t,t[np.isnan(kappa)])
kappa = kappa[t]
X_nodes = X_nodes[t]
t = np.arange(X_nodes.shape[0])
kappa = np.array(kappa)
# kappa_diff = np.exp(kappa[:,None] - kappa[:,None].T)

import torch
from torch_geometric.nn import knn_graph

pos = [list(X_nodes[i]) for i in t]
pos = torch.tensor(pos, dtype=torch.float)
node_feature = [list(np.hstack([X_nodes[i],kappa[i]])) for i in t]
node_feature = torch.tensor(node_feature, dtype=torch.float)
kappa = list(kappa)
kappa = torch.tensor(kappa, dtype=torch.float)

edge_index = knn_graph(node_feature, k=6)
from torch_geometric.utils import to_undirected
edge_index = to_undirected(edge_index, len(kappa))

from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
train_id, test_id = train_test_split(np.arange(len(kappa)), test_size=0.2, random_state=0)
test_id, val_id = train_test_split(test_id, test_size=0.5, random_state=0)
train_mask = np.zeros(len(kappa), dtype=bool)
test_mask = np.zeros(len(kappa), dtype=bool)
val_mask = np.zeros(len(kappa), dtype=bool)
train_mask[train_id] = True
test_mask[test_id] = True
val_mask[val_id] = True

data = Data(x=pos, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
# data = Data(x=kappa[:,None], edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
torch.save(data,'data.pt')

# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

# from torch_geometric.utils.convert import to_networkx

# G = to_networkx(data, node_attrs=['kappa','pos','x'], edge_attrs=None, to_undirected=False,
#                 remove_self_loops=True)
# plot.plot_graph(G,node_colors=kappa.numpy(),show_colorbar=False,ax=ax2,node_size=5,edge_width=0.5)
# plot.plot_graph(G,node_colors=kappa.numpy(),show_colorbar=False,layout='spectral',node_size=5,edge_width=0.5)


