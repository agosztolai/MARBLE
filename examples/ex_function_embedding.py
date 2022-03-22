#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from GeoDySys.plotting import set_colors
from GeoDySys import plotting, embedding
from GeoDySys.embedding import SAGE, model_eval, train
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data.collate import collate
from tensorboardX import SummaryWriter
from datetime import datetime
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from scipy.sparse import coo_array
import scipy.sparse as scp

from sklearn.cluster import KMeans

def main():

    n = 200
    k = 10
    include_position_in_feature = False
    n_clusters=10
    #training parameters
    par = {'hidden_channels': 8,
           'batch_size': 800,
           'num_layers': 1,
           'n_neighbours': [10], #parameter of neighbourhood sampling
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
    
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    #set up model
    model = SAGE(in_channels=data_train.x.shape[1], 
                 hidden_channels=par['hidden_channels'], 
                 num_layers=par['num_layers'])
    
    #train
    model = train(model, data_train, par, writer)
    
    #embed data and cluster
    emb = model_eval(model, data_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    labels = kmeans.labels_
    
    counts = []
    for i in range(len(slices['x'])-1):
        counts.append(labels[slices['x'][i]:slices['x'][i+1]])
    
    #plot
    model_vis(emb, data_train) #TSNE embedding
    
    #histograms
    fig, axs = plt.subplots(2,2)   
    axs[0,0].hist(counts[0])
    axs[1,0].hist(counts[1])
    axs[1,1].hist(counts[2])
    axs[0,1].hist(counts[3])
    fig.tight_layout()
    
    #sampled functions
    fig, axs = plt.subplots(2,2)
       
    c= set_colors(y[0], cbar=False)
    axs[0,0].scatter(x[0][:,0],x[0][:,1], c=c)
    axs[0,0].set_aspect('equal', 'box')
    plotting.graph(G[0],node_colors=None,show_colorbar=False,ax=axs[0,0],node_size=5,edge_width=0.5)

    c= set_colors(y[1], cbar=False)
    axs[1,0].scatter(x[1][:,0],x[1][:,1], c=c)
    axs[1,0].set_aspect('equal', 'box')
    plotting.graph(G[1],node_colors=None,show_colorbar=False,ax=axs[1,0],node_size=5,edge_width=0.5)
    
    c= set_colors(y[2], cbar=False)
    axs[1,1].scatter(x[2][:,0],x[2][:,1], c=c)
    axs[1,1].set_aspect('equal', 'box')
    plotting.graph(G[2],node_colors=None,show_colorbar=False,ax=axs[1,1],node_size=5,edge_width=0.5)
    
    c= set_colors(y[3], cbar=False)
    axs[0,1].scatter(x[3][:,0],x[3][:,1], c=c)
    axs[0,1].set_aspect('equal', 'box')
    plotting.graph(G[3],node_colors=None,show_colorbar=False,ax=axs[0,1],node_size=5,edge_width=0.5)
    
    fig.tight_layout()
    

def model_vis(emb, data):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    colors = ["red", "orange", "green", "blue", "purple", "brown", "black"]
    colors = [colors[y] for y in data.y]
    if emb.shape[1]>2:
        xs, ys = zip(*TSNE().fit_transform(emb.detach().numpy()))
    else:
        xs, ys = emb[:,0], emb[:,1]
    plt.scatter(xs, ys, color=colors, alpha=0.3)
    
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
    u = u - torch.swapaxes(u,0,1)
    A = to_scipy_sparse_matrix(data.edge_index)
    ind = scp.find(A)
    mask=torch.tensor(A.todense(),dtype=bool)
    mask=mask[:,:,None].repeat(1,1,2)
    u[~mask] = 0
    u = u.numpy()
    
    F = []
    for g in gauge:
        g = np.array(g)[None]
        g=np.repeat(g,n,axis=0)
        g=g[:,None]
        g=np.repeat(g,n,axis=1)
        
        _F = np.zeros([n,n])
        for i,j in zip(ind[0],ind[1]):
            _F[i,j] = g[i,j,:].dot(u[i,j,:])
        F.append(_F)
        
    return torch.tensor(sum(F))


def aggr_directional_derivative(F):
    EPS = 1e-8
    Fhat = F / (torch.sum(torch.abs(F), keepdim=True, dim=1) + EPS)
    Bdx = Fhat - torch.diag(torch.sum(Fhat, dim=1))
    
    return Bdx


if __name__ == '__main__':
    sys.exit(main())