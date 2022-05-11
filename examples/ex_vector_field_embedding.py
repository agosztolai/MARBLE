#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from torch_geometric import seed
from GeoDySys import plotting, utils
from GeoDySys.model import net


def main():
    
    seed.seed_everything(1)
    
    #parameters
    n = 500
    k = 30
    n_clusters = 15
    
    par = {'batch_size': 400, #batch size, this should be as large as possible
           'epochs': 30, #optimisation epochs
           'n_conv_layers': 1, #number of hops in neighbourhood
           'n_lin_layers': 3,
           'hidden_channels': 32, #number of internal dimensions in MLP
           'out_channels': 8,
           'n_neighbours': k, #parameter of neighbourhood sampling
           'dropout': 0.3, #dropout in MLP
           'adj_norm':True
           }
      
    #evaluate functions
    # f0: linear, f1: point source, f2: point vortex, f3: doublet
    x0 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x1 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x2 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2)) 
    x3 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x = [x0, x1, x2, x3]
    # y = [f0(x0)[:,[1]], f1(x1)[:,[1]], f2(x2)[:,[1]], f3(x3)[:,[1]]] #evaluated functions
    y = [f0(x0), f1(x1), f2(x2), f3(x3)] #evaluated functions
        
    #construct PyG data object
    data_train, slices, G = utils.construct_data_object(x, y, graph_type='knn', k=k)
    
    #train model
    model = net(data_train, kernel='directional_derivative', gauge='global', **par)
    model.train_model(data_train)
    emb = model.eval_model(data_train)
    clusters = utils.cluster(emb, n_clusters=n_clusters)
    
    #plot
    titles=['Linear','Point source','Point vortex','Saddle']
    plot_functions(y, G, titles=titles) #sampled functions
    plotting.embedding(emb, clusters, data_train.y.numpy(), titles=titles) #TSNE embedding 
    plotting.histograms(clusters, slices, titles=titles) #histograms
    plotting.neighbourhoods(G, y, clusters, n_samples=4, vector=True) #neighbourhoods
    
    
def f0(x):
    f = x*0 + 1
    return torch.tensor(f).float()

def f1(x):
    eps = 1e-3
    norm = np.sqrt(x[:,[0]]**2 + x[:,[1]]**2) + eps
    u = 2*x[:,[0]]/norm
    v = 2*x[:,[1]]/norm
    return torch.tensor(np.hstack([u,v])).float()

def f2(x):
    eps = 1e-1
    norm = x[:,[0]]**2 + x[:,[1]]**2 + eps
    u = x[:,[1]]/norm
    v = -x[:,[0]]/norm
    return torch.tensor(np.hstack([u,v])).float()

def f3(x):
    eps = 1e-1
    norm = x[:,[0]]**2 + x[:,[1]]**2 + eps
    u = x[:,[1]]/norm
    v = x[:,[0]]/norm
    return torch.tensor(np.hstack([u,v])).float()


def plot_functions(y, graphs, titles=None):
    import matplotlib.gridspec as gridspec
    import networkx as nx
    fig = plt.figure(figsize=(10,10), constrained_layout=True)
    grid = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    
    for i, (_y, G) in enumerate(zip(y,graphs)):
        ax = plt.Subplot(fig, grid[i])
        ax.set_aspect('equal', 'box')
        c=plotting.set_colors(_y.numpy(), cbar=False)
        # plotting.graph(G,node_values=c,show_colorbar=False,ax=ax,node_size=30,edge_width=0.5)
        
        plotting.graph(G,node_values=None,show_colorbar=False,ax=ax,node_size=30,edge_width=0.5)
        x = np.array(list(nx.get_node_attributes(G,name='x').values()))
        c = np.array(np.sqrt(_y[:,0]**2 + _y[:,1]**2))
        c = plotting.set_colors(c, cbar=False)
        ax.quiver(x[:,0],x[:,1],_y[:,0],_y[:,1], color=c, scale=10, scale_units='x',width=0.005)
        
        if titles is not None:
            ax.set_title(titles[i])
        fig.add_subplot(ax)  


if __name__ == '__main__':
    sys.exit(main())