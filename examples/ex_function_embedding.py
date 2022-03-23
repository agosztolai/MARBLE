#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from GeoDySys import plotting, embedding
from GeoDySys.embedding import SAGE, model_eval, train
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data.collate import collate
from tensorboardX import SummaryWriter
from datetime import datetime
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import scipy.sparse as scp

from sklearn.cluster import KMeans

def main():

    n = 1000
    k = 10
    include_position_in_feature = False
    n_clusters=10
    #training parameters
    par = {'hidden_channels': 8,
           'batch_size': 800,
           'num_layers': 2,
           'n_neighbours': 10, #parameter of neighbourhood sampling
           'epochs': 100,
           'lr': 0.01,
           'edge_dropout':0.0}
    
    ind = np.arange(n)
      
    x0 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x1 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x2 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2)) 
    x3 = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
    x = [x0, x1, x2, x3]
    y = [f0(x0), f1(x1), f2(x2), f3(x3)]
    
    data_list = []
    G = []
    for i, y_ in enumerate(y):
        #fit knn before adding function as node attribute
        data_ = embedding.fit_knn_graph(x[i], ind, k=k)
        data_.pos = torch.tensor(x[i])
            
        #save graph for testing and plotting
        G_ = to_networkx(data_, node_attrs=['x'], edge_attrs=None, to_undirected=True,
                remove_self_loops=True)
            
        #add new node feature
        if include_position_in_feature:
            data_.x = torch.hstack((data_.x,y_)) #include positional features
        else:
            data_.x = y_ #only function value as feature

        data_.num_nodes = len(x[i])
        data_.num_node_features = data_.x.shape[1]
        data_.y = torch.ones(data_.num_nodes, dtype=int)*i
        data_ = embedding.traintestsplit(data_, test_size=0.1, val_size=0.5, seed=0)
        
        G.append(G_)
        data_list.append(data_)
        
    #collate datasets
    data_train, slices, _ = collate(data_list[0].__class__,
                                    data_list=data_list,
                                    increment=True,
                                    add_batch=False)
    
    #define kernels
    F = project_gauge_to_neighbours(data_train, [[1,0],[0,1]], local=False)
    data_train.kernels = aggr_directional_derivative(F)
    
    #set up model
    model = SAGE(in_channels=data_train.x.shape[1], 
                 hidden_channels=par['hidden_channels'], 
                 num_layers=par['num_layers'])
    
    #train
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = train(model, data_train, par, writer)
    
    #embed data and cluster
    emb = model_eval(model, data_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    labels = kmeans.labels_
    
    #plot
    plotting.embedding(emb, data_train, labels=['Constant','Linear','Parabola','Saddle']) #TSNE embedding 
    
    #sampled functions
    fig, axs = plt.subplots(2,2)       
    axs[0,0].set_aspect('equal', 'box')
    plotting.graph(G[0],node_colors=y[0],show_colorbar=False,ax=axs[0,0],node_size=5,edge_width=0.5)
    axs[0,1].set_aspect('equal', 'box')
    plotting.graph(G[1],node_colors=y[1],show_colorbar=False,ax=axs[1,0],node_size=5,edge_width=0.5)    
    axs[1,0].set_aspect('equal', 'box')
    plotting.graph(G[2],node_colors=y[2],show_colorbar=False,ax=axs[1,1],node_size=5,edge_width=0.5)    
    axs[1,1].set_aspect('equal', 'box')
    plotting.graph(G[3],node_colors=y[3],show_colorbar=False,ax=axs[0,1],node_size=5,edge_width=0.5)   
    fig.tight_layout()
    
    #histograms
    plotting.histograms(labels, slices, titles=['Constant','Linear','Parabola','Saddle'])
    
    #neighbourhoods
    n_samples = 4
    plotting.neighbourhoods(G, y, n_clusters, n_samples, labels, n)
    
    
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


def project_gauge_to_neighbours(data, gauge, local=False):
    n = len(data.x)
    u = data.pos[:,None].repeat(1,n,1)
    u = torch.swapaxes(u,0,1) - u #uij = xj - xi 
    A = to_scipy_sparse_matrix(data.edge_index)
    
    mask = torch.tensor(A.todense(), dtype=bool)
    mask = mask[:,:,None].repeat(1,1,2)
    u[~mask] = 0
    u = u.numpy()
    
    F = []
    for g in gauge:
        g = np.array(g)[None]
        g = np.repeat(g,n,axis=0)
        g = g[:,None]
        g = np.repeat(g,n,axis=1)
        
        _F = np.zeros([n,n])
        ind = scp.find(A)
        for i,j in zip(ind[0], ind[1]):
            _F[i,j] = g[i,j,:].dot(u[i,j,:])
        F.append(_F)
        
    return torch.tensor(sum(F)/len(F))


def aggr_directional_derivative(F):
    EPS = 1e-8
    Fhat = F / (torch.sum(torch.abs(F), keepdim=True, dim=1) + EPS)
    Bdx = Fhat - torch.diag(torch.sum(Fhat, dim=1))
    
    return Bdx


if __name__ == '__main__':
    sys.exit(main())