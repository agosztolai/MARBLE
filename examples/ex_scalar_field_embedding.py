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
    n_clusters = 15
    
    par = {'batch_size': 400, #batch size, this should be as large as possible
           'epochs': 20, #optimisation epochs
           'n_conv_layers': 1, #number of hops in neighbourhood
           'hidden_channels': 8, #number of internal dimensions in MLP 
           'n_neighbours': k, #parameter of neighbourhood sampling
           'b_norm': False, #batch norm
           'dropout': 0.3, #dropout in MLP
           'adj_norm': True
           }
      
    #evaluate functions
    # f1: constant, f2: linear, f3: parabola, f4: saddle
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
    model.train_model(data_train)
    emb = model.eval_model(data_train)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    labels = kmeans.labels_
    
    #plot
    titles=['Constant','Linear','Parabola','Saddle']
    plot_functions(y, G, titles=titles) #sampled functions
    plotting.embedding(emb, kmeans, data_train.y.numpy(), titles=titles) #TSNE embedding 
    plotting.histograms(labels, slices, titles=titles) #histograms
    plotting.neighbourhoods(G, y, n_clusters, labels, n_samples=4, norm=True) #neighbourhoods
    
    
def f0(x):
    f = x[:,[0]]*0
    return torch.tensor(f).float()

def f1(x):
    f = x[:,[0]] + x[:,[1]]
    return torch.tensor(f).float()

def f2(x):
    f = x[:,[0]]**2 + x[:,[1]]**2
    return torch.tensor(f).float()

def f3(x):
    f = x[:,[0]]**2 - x[:,[1]]**2
    return torch.tensor(f).float()


def plot_functions(y, graphs, titles=None):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10,10), constrained_layout=True)
    grid = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    
    for i, (_y, G) in enumerate(zip(y,graphs)):
        ax = plt.Subplot(fig, grid[i])
        ax.set_aspect('equal', 'box')
        c=plotting.set_colors(_y.numpy(), cbar=False)
        plotting.graph(G,node_values=c,show_colorbar=False,ax=ax,node_size=30,edge_width=0.5)
        
        if titles is not None:
            ax.set_title(titles[i])
        fig.add_subplot(ax)  


if __name__ == '__main__':
    sys.exit(main())