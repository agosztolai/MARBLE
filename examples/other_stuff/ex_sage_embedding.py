#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from GeoDySys import solvers, curvature, plotting, time_series
import networkx as nx
import torch
from torch_geometric.nn import knn_graph
            

"""1. Simulate system"""
# x0 = [0.1,0.1] 
par = {'mu': 1.}
fun = 'vanderpol'

n=100
# x0 = [-8.0, 7.0]
X0_range = [[-5,5],[-5,5]]
t = np.linspace(0, 20, n)
t_ind = np.arange(n)
mu, sigma = 0, .01 # mean and standard deviation
t_ind, X = solvers.generate_trajectories(fun, n, t, X0_range, par=par, seed=0, transient=0.1, stack=True)#, mu=mu, sigma=sigma)
# X = solvers.simulate_ODE(fun, t, x0, par, noise=False, mu=mu, sigma=sigma)
t_sample = t_ind


"""2. Random project and then delay embed"""
# x = time_series.random_projection(X, seed=0)
# dim = 3
# X = time_series.delay_embed(X[:,0],dim,tau=-1)
# t_sample = t_sample[:-dim]


"""3. Compute curvature of trajectories starting at t_sample"""
# n_sample=200
# t_sample = random.sample(list(np.arange(n)), n_sample)

T=5
kappas = curvature.curvature_trajectory(X,t_ind,t_sample,T)
# kappas = np.clip(kappas, -0.1, 0.1)


"""4. Plotting"""

ax = plotting.trajectories(X, node_feature=kappas, style='o', lw=1, ms=1,alpha=1)

dist, nn = time_series.find_nn(X[t_sample], X, nn=10, nmax=20)
t_nn = np.hstack([np.array(t_sample)[:,None],np.array(nn)])
ts, tt = time_series.valid_flows(t_ind, t_nn.flatten(), T)
ts = ts.reshape(t_nn.shape)
tt = tt.reshape(t_nn.shape)
ax = plotting.trajectories(X, node_feature=None, style='o', lw=1, ms=1)
flows_n, _, _ = time_series.generate_flow(X, ts[20,1:], T)
plotting.trajectories(np.vstack(flows_n), ax=ax, node_feature='C1', style='o', lw=1, ms=4)
flow, _, _ = time_series.generate_flow(X, ts[20,[0]], T)
plotting.trajectories(np.vstack(flow), ax=ax, node_feature='C3', style='o', lw=1, ms=4)



"""4. Train GNN"""


#form nn graph
# numn=10

# X_nodes = X[t]
# t = np.arange(n)
# t = np.delete(t,t[np.isnan(kappa)])
# kappa = kappa[t]
# X_nodes = X_nodes[t]
# t = np.arange(X_nodes.shape[0])
# kappa = np.array(kappa)
# kappa_diff = np.exp(kappa[:,None] - kappa[:,None].T)



# pos = [list(X_nodes[i]) for i in t]
# pos = torch.tensor(pos, dtype=torch.float)
# node_feature = [list(np.hstack([X_nodes[i],kappa[i]])) for i in t]
# node_feature = torch.tensor(node_feature, dtype=torch.float)
# kappa = list(kappa)
# kappa = torch.tensor(kappa, dtype=torch.float)

# edge_index = knn_graph(node_feature, k=6)
# from torch_geometric.utils import to_undirected
# edge_index = to_undirected(edge_index, len(kappa))

# from torch_geometric.data import Data
# from sklearn.model_selection import train_test_split
# train_id, test_id = train_test_split(np.arange(len(kappa)), test_size=0.2, random_state=0)
# test_id, val_id = train_test_split(test_id, test_size=0.5, random_state=0)
# train_mask = np.zeros(len(kappa), dtype=bool)
# test_mask = np.zeros(len(kappa), dtype=bool)
# val_mask = np.zeros(len(kappa), dtype=bool)
# train_mask[train_id] = True
# test_mask[test_id] = True
# val_mask[val_id] = True

# data = Data(x=pos, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)




# data = Data(x=kappa[:,None], edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
# torch.save(data,'data.pt')

# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

# from torch_geometric.utils.convert import to_networkx

# G = to_networkx(data, node_attrs=['kappa','pos','x'], edge_attrs=None, to_undirected=False,
#                 remove_self_loops=True)
# plot.plot_graph(G,node_colors=kappa.numpy(),show_colorbar=False,ax=ax2,node_size=5,edge_width=0.5)
# plot.plot_graph(G,node_colors=kappa.numpy(),show_colorbar=False,layout='spectral',node_size=5,edge_width=0.5)


