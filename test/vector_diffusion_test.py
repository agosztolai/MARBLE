#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
from GeoDySys import plotting, utils, geometry
from GeoDySys.layers import Diffusion
import torch


def main():
    
    #parameters
    n = 512
    k = 30
    tau = 1000
    
    # f1: constant, f2: linear, f3: parabola, f4: saddle
    x3 = geometry.sample_2d(n, [[-1,-1],[1,1]], 'random')
    x = [x3]
    y = [f3(x3)] #evaluated functions
        
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
    
    gauges, R = geometry.compute_gauges(data, False, 50)
    L = geometry.compute_laplacian(data)
    Lc = geometry.compute_connection_laplacian(data, R)
    
    diffusion = Diffusion(L=L, Lc=Lc)
    diffusion.diffusion_time.data = torch.tensor(tau)
    data.x = diffusion(data.x, vector=True, normalize=True)
 
    #plot
    titles=['Vortex left']
    plot_functions(data, titles=titles) #sampled functions


def f3(x):
    eps = 1e-1
    norm = np.sqrt((x[:,[0]]-1)**2 + x[:,[1]]**2 + eps)
    u = x[:,[1]]/norm
    v = -(x[:,[0]]-1)/norm
    return np.hstack([u,v])


def plot_functions(data, titles=None):
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx
    fig = plt.figure(figsize=(10,10), constrained_layout=True)
    grid = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    
    data_list = data.to_data_list()
    
    for i, d in enumerate(data_list):
        ax = plt.Subplot(fig, grid[i])
        ax.set_aspect('equal', 'box')
        
        G = to_networkx(d, node_attrs=['pos'], edge_attrs=None, to_undirected=True,
                remove_self_loops=True)
        
        plotting.graph(G,node_values=None,show_colorbar=False,ax=ax,node_size=30,edge_width=0.5)
        x = np.array(list(nx.get_node_attributes(G,name='pos').values()))
        c = np.array(np.sqrt(d.x[:,0]**2 + d.x[:,1]**2))
        c = plotting.set_colors(c, cbar=False)
        ax.quiver(x[:,0],x[:,1],d.x[:,0],d.x[:,1], color=c, scale=10, scale_units='x',width=0.005)
        
        if titles is not None:
            ax.set_title(titles[i])
        fig.add_subplot(ax)  


if __name__ == '__main__':
    sys.exit(main())