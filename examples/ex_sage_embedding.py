#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from GeoDySys import solvers, curvature, plotting, time_series, embedding
import networkx as nx
import sys
import random
from torch_geometric.utils.convert import to_networkx
            
def main():
    """1. Simulate system"""
    # par = {'beta': 0.4, 'sigma': -1}
    # fun = 'hopf'
    
    par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
    fun = 'lorenz'
    
    ntraj=5
    tn=1000
    X0_range = [[-5,5],[-5,5],[-5,5]]
    t = np.linspace(0, 10, tn)
    t_ind = np.arange(tn)
    mu, sigma = 0, .0 # mean and standard deviation of additive noise
    t_ind, X = solvers.generate_trajectories(fun, ntraj, t, X0_range, par=par, seed=0, transient=0.1, stack=True, mu=mu, sigma=sigma)
    t_sample = t_ind
    
    
    """2. Random project and then delay embed"""
    # x = time_series.random_projection(X, seed=0)
    # dim = 3
    # X = time_series.delay_embed(X[:,0],dim,tau=-1)
    # t_sample = t_sample[:-dim]
    
    
    """3. Compute curvature of trajectories starting at t_sample"""
    n_sample=200
    t_sample = random.sample(list(np.arange(len(t_sample))), n_sample)
    
    T=5
    # kappas = curvature.curvature_trajectory(X,t_ind,t_sample,T,nn=10)
    # kappas = np.clip(kappas, -0.1, 0.1)
    
    
    """4. Train GNN"""
    
    # X = X[~kappas.mask]
    # kappa = kappa[~kappas.mask]
    
    # t = np.arange(X_nodes.shape[0])
    # kappa = np.array(kappa)
    
    data = embedding.fit_knn_graph(X, t_sample, k=2)
    
    G = to_networkx(data, node_attrs=['x'], edge_attrs=None, to_undirected=False,
                    remove_self_loops=True)
    
    ax = plotting.trajectories(X, node_feature=None, style='o', lw=1, ms=1, alpha=1,axis=True)
    plotting.graph(G,node_colors=None,show_colorbar=False,ax=ax,node_size=5,edge_width=0.5)
    
    #spectral embedding
    # plotting.graph(G,node_colors=None,show_colorbar=False,layout='spectral',node_size=5,edge_width=0.5)

    data = embedding.traintestsplit(data, test_size=0.1, val_size=0.5, seed=0)
    # torch.save(data,'data.pt')
    
    
if __name__ == '__main__':
    sys.exit(main())

