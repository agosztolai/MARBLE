#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
from GeoDySys import plotting, utils, geometry
from GeoDySys.main import net


def main():
    
    #parameters
    n = 512
    k = 30
    n_clusters = 10
    
    par = {'batch_size': 256, #batch size, this should be as large as possible
           'epochs': 20, #optimisation epochs
           'order': 1, #order of derivatives
           'n_lin_layers': 2,
           'hidden_channels': 16, #number of internal dimensions in MLP
           'out_channels': 8,
           }
      
    #evaluate functions
    # f0: linear, f1: point source, f2: point vortex, f3: saddle
    x0 = geometry.sample_2d(n, [[-1,-1],[1,1]], 'random')
    x1 = geometry.sample_2d(n, [[-1,-1],[1,1]], 'random')
    x2 = geometry.sample_2d(n, [[-1,-1],[1,1]], 'random')
    x3 = geometry.sample_2d(n, [[-1,-1],[1,1]], 'random')
    x = [x0, x1, x2, x3]
    # y = [f0(x0)[:,[1]], f1(x1)[:,[1]], f2(x2)[:,[1]], f3(x3)[:,[1]]] #evaluated functions
    y = [f0(x0), f1(x1), f2(x2), f3(x3)] #evaluated functions
        
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
    
    #train model
    model = net(data, gauge='global', **par)
    model.train_model(data)
    emb = model.evaluate(data)
    emb, clusters = geometry.cluster_and_embed(emb, n_clusters=n_clusters)
    
    #plot
    titles=['Linear left','Linear right','Vortex right','Vortex left']
    plot_functions(data, titles=titles) #sampled functions
    plt.savefig('../results/vector_fields.svg')
    plotting.embedding(emb, clusters, data.y.numpy(), titles=titles) #TSNE embedding 
    plt.savefig('../results/vector_fields_embedding.svg')
    plotting.histograms(data, clusters, titles=titles) #histograms
    plt.savefig('../results/vector_fields_embedding.svg')
    plotting.neighbourhoods(data, clusters, n_samples=4, vector=True) #neighbourhoods
    plt.savefig('../results/vector_fields_nhoods.svg')
    
def f0(x):
    return x*0 + np.array([-1,-1])

def f1(x):
    return x*0 + np.array([1,1])

# def f2(x):
#     return x*0 + np.array([1,0])

def f3(x):
    return x*0 + np.array([0,1])

# def f1(x):
#     eps = 1e-3
#     norm = np.sqrt(x[:,[0]]**2 + x[:,[1]]**2) + eps
#     u = 2*x[:,[0]]/norm
#     v = 2*x[:,[1]]/norm
#     return np.hstack([u,v])

def f2(x):
    eps = 1e-1
    norm = np.sqrt((x[:,[0]]+1)**2 + x[:,[1]]**2 + eps)
    u = x[:,[1]]/norm
    v = -(x[:,[0]]+1)/norm
    return np.hstack([u,v])

def f3(x):
    eps = 1e-1
    norm = np.sqrt((x[:,[0]]-1)**2 + x[:,[1]]**2 + eps)
    u = x[:,[1]]/norm
    v = -(x[:,[0]]-1)/norm
    return np.hstack([u,v])

# def f3(x):
#     eps = 1e-1
#     norm = x[:,[0]]**2 + x[:,[1]]**2 + eps
#     u = x[:,[1]]/norm
#     v = x[:,[0]]/norm
#     return torch.tensor(np.hstack([u,v])).float()


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