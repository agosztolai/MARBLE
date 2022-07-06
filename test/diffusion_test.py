#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import sys
from GeoDySys import plotting, utils, geometry
from GeoDySys.layers import Diffusion
import torch


def main():
    
    #parameters
    n = 512
    k = 30
    
    # f1: constant, f2: linear, f3: parabola, f4: saddle
    x3 = geometry.sample_2d(n, [[-1,-1],[1,1]], 'random')
    x = [x3]
    y = [f3(x3)] #evaluated functions
        
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
    
    L = geometry.compute_laplacian(data, k_eig=128, eps = 1e-8)
    diffusion = Diffusion(data.x.shape[1], method='matrix_exp')
    diffusion.diffusion_time[0].data = torch.tensor(4)
    data.x = diffusion(data.x, L)#.detach()
 
    #plot
    titles=['Constant','Linear','Parabola','Saddle']
    plot_functions(data, titles=titles) #sampled functions
    
    
def f0(x):
    return x[:,[0]]*0

def f1(x):
    return x[:,[0]] + x[:,[1]]

def f2(x):
    return x[:,[0]]**2 + x[:,[1]]**2

def f3(x):
    return x[:,[0]]**2 - x[:,[1]]**2


def plot_functions(data, titles=None):
    import matplotlib.gridspec as gridspec
    from torch_geometric.utils.convert import to_networkx

    fig = plt.figure(figsize=(10,10), constrained_layout=True)
    grid = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    
    data_list = data.to_data_list()
    
    for i, d in enumerate(data_list):
        
        G = to_networkx(d, node_attrs=['pos'], edge_attrs=None, to_undirected=True,
                remove_self_loops=True)
        
        ax = plt.Subplot(fig, grid[i])
        ax.set_aspect('equal', 'box')
        c=plotting.set_colors(d.x.numpy(), cbar=False)
        plotting.graph(G,node_values=c,show_colorbar=False,ax=ax,node_size=30,edge_width=0.5)
        
        if titles is not None:
            ax.set_title(titles[i])
        fig.add_subplot(ax)  


if __name__ == '__main__':
    sys.exit(main())