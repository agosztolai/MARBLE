#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from torch_geometric import seed
from GeoDySys import plotting, utils
from GeoDySys.model import net
from sklearn.cluster import KMeans



def main():
    
    seed.seed_everything(1)
    
    #parameters
    n = 500
    k = 30
    n_clusters = 20
    
    par = {'batch_size': 100, #batch size, this should be as large as possible
           'epochs': 10, #optimisation epochs
           'n_conv_layers': 1, #number of hops in neighbourhood
           'n_lin_layers': 2, #number of layers if MLP
           'hidden_channels': 8, #number of internal dimensions in MLP 
           'n_neighbours': k, #parameter of neighbourhood sampling
           'lr': 0.01, #learning rate
           'b_norm': False, #batch norm
           'adj_norm': True, #adjacency-wise norm
           'activation': True, #relu
           'dropout': 0.3, #dropout in MLP
           'edge_dropout':0.0 #dropout in convolution graph
           }
      
    #evaluate functions
    # f0: linear, f1: point source, f2: point vortex, f3: doublet
    x0 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x1 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x2 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2)) 
    x3 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x = [x0, x1, x2, x3]
    y = [f0(x0), f1(x1), f2(x2), f3(x3)] #evaluated functions
        
    #construct PyG data object
    data_train, slices, G = utils.construct_data_object(x, y, graph_type='knn', k=k)
    
    #set up model
    model = net(data_train, kernel='directional_derivative', gauge='global', **par)
    model.train_model(data_train, par)
    emb = model.eval_model(data_train)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    labels = kmeans.labels_
    
    #plot
    titles=['Random','Linear','Parabola','Saddle']
    plot_functions(y, G, titles=titles) #sampled functions
    plotting.embedding(emb, kmeans, data_train.y.numpy(), titles=titles) #TSNE embedding 
    plotting.histograms(labels, slices, titles=titles) #histograms
    plotting.neighbourhoods(G, y, n_clusters, labels, n_samples=4, vector=True) #neighbourhoods
    
    
def f0(x):
    f = x*0 + 1
    return torch.tensor(f).float()

def f1(x):
    norm = x[:,[0]]**2 + x[:,[1]]**2
    u = 2*x[:,[0]]/norm
    v = 2*x[:,[1]]/norm
    return torch.tensor(np.hstack([u,v])).float()

def f2(x):
    norm = 1 + (x[:,[1]]/x[:,[0]])**2
    u = -(x[:,[1]]/x[:,[0]]**2)/norm
    v = (1/x[:,[0]])/norm
    return torch.tensor(np.hstack([u,v])).float()

def f3(x):

    u = np.sin(x[:,[0]])*np.cos(x[:,[1]])
    v = -np.cos(x[:,[0]])*np.sin(x[:,[1]])
    return torch.tensor(np.hstack([u,v])).float()


def plot_functions(y, graphs, titles=None):
    import matplotlib.gridspec as gridspec
    import networkx as nx
    fig = plt.figure(figsize=(10,10), constrained_layout=True)
    grid = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    
    for i, (_y, G) in enumerate(zip(y,graphs)):
        ax = plt.Subplot(fig, grid[i])
        ax.set_aspect('equal', 'box')
        # c=plotting.set_colors(_y.numpy(), cbar=False)
        plotting.graph(G,node_colors=None,show_colorbar=False,ax=ax,node_size=30,edge_width=0.5)
        x = np.array(list(nx.get_node_attributes(G,name='x').values()))
        ax.quiver(x[:,0],x[:,1],_y[:,0],_y[:,1], color='b')
        
        if titles is not None:
            ax.set_title(titles[i])
        fig.add_subplot(ax)  


if __name__ == '__main__':
    sys.exit(main())