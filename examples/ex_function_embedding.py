#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from torch_geometric import seed
from GeoDySys import plotting, utils, kernels
from GeoDySys.model import net
from GeoDySys.traintest import model_eval, train, split
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data.collate import collate
from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn.cluster import KMeans



def main():
    
    seed.seed_everything(2)
    
    #parameters
    n = 200
    k = 30
    n_clusters = 30
    
    par = {'batch_size': 100, #batch size, this should be as large as possible
           'epochs': 50, #optimisation epochs
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
    x0 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x1 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x2 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2)) 
    x3 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x = [x0, x1, x2, x3]
    y = [f0(x0), f1(x1), f2(x2), f3(x3)] #evaluated functions
    
    ind = np.arange(n) #take all points in functions, layer we may change if we subsample
    
    data_list = []
    G = []
    for i, y_ in enumerate(y):
        #fit knn before adding function as node attribute
        data_ = utils.fit_knn_graph(x[i], ind, k=k)
        data_.pos = torch.tensor(x[i])
            
        #save graph for testing and plotting
        G_ = to_networkx(data_, node_attrs=['x'], edge_attrs=None, to_undirected=True,
                remove_self_loops=True)
            
        #build pytorch geometric object
        data_.x = y_ #only function value as feature
        data_.num_nodes = len(x[i])
        data_.num_node_features = data_.x.shape[1]
        data_.y = torch.ones(data_.num_nodes, dtype=int)*i
        data_ = split(data_, test_size=0.1, val_size=0.5, seed=0)
        
        G.append(G_)
        data_list.append(data_)
        
    #collate datasets
    data_train, slices, _ = collate(data_list[0].__class__,
                                    data_list=data_list,
                                    increment=True,
                                    add_batch=False)
    
    #define kernels
    gauge = [[1,0],[0,1]]
    K = kernels.aggr_directional_derivative(data_train, gauge)
    data_train.kernels = K
    
    #set up model
    model = net(in_channels=len(gauge)*data_train.x.shape[1], **par)
    
    #train
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = train(model, data_train, par, writer)
    
    #embed data
    emb = model_eval(model, data_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    labels = kmeans.labels_
    
    #plot
    titles=['Constant','Linear','Parabola','Saddle']
    
    #sampled functions
    plot_functions(y, G, titles=titles)
    
    #embedding
    plotting.embedding(emb, kmeans, data_train.y.numpy(), titles=titles) #TSNE embedding 
    
    #histograms
    plotting.histograms(labels, slices, titles=titles)
    
    #neighbourhoods
    n_samples = 4
    plotting.neighbourhoods(G, y, n_clusters, n_samples, labels, norm=False)
    
    
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
        plotting.graph(G,node_colors=c,show_colorbar=False,ax=ax,node_size=30,edge_width=0.5)
        
        if titles is not None:
            ax.set_title(titles[i])
        fig.add_subplot(ax)  


if __name__ == '__main__':
    sys.exit(main())