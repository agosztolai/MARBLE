#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import networkx as nx
import numpy as np

from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, Batch
from torch_sparse import SparseTensor

from cknn import cknneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def adjacency_matrix(edge_index, size=None, value=None):
    """Compute adjacency matrix from edge_index"""
    if value is not None:
        value=value[edge_index[0], edge_index[1]]
    if size is None:
        size = (edge_index.max()+1, edge_index.max()+1)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                       value=value,
                       sparse_sizes=(size[0], size[1]))
    
    return adj


def construct_dataset(x, y=None, graph_type='cknn', k=10):
    """Construct PyG dataset from node positions and features"""
        
    if not isinstance(x, (list,tuple)):
        x = [x]
    if not isinstance(y, (list,tuple)):
        y = [y]
        
    data_list = []
    for i, y_ in enumerate(y):
        #fit knn before adding function as node attribute
        x_ = torch.tensor(x[i], dtype=torch.float)
        edge_index = fit_graph(x_, graph_type=graph_type, par=k)
        data_ = Data(x=x_, edge_index=edge_index)
        data_.pos = torch.tensor(x[i])
        if y_ is None:
            A = adjacency_matrix(edge_index)
            y_ = A.sum(1).unsqueeze(-1)
        data_.x = torch.tensor(y_).float() #only function value as feature
        data_.num_nodes = len(x[i])
        data_.num_node_features = data_.x.shape[1]
        data_.y = torch.ones(data_.num_nodes, dtype=int)*i
        
        data_list.append(data_)
        
    #collate datasets
    batch = Batch.from_data_list(data_list)
    
    #split into training/validation/test datasets
    split = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    split(batch)
    
    return batch


def fit_graph(X, graph_type='cknn', par=1):
    """Fit graph to node positions"""
    
    ckng = cknneighbors_graph(X, n_neighbors=par, delta=1.0)
    
    if graph_type=='cknn':
        edge_index = torch.tensor(list(nx.Graph(ckng).edges), dtype=torch.int64).T
    elif graph_type=='knn':
        edge_index = knn_graph(X, k=par)
    elif graph_type=='radius':
        edge_index = radius_graph(X, r=par)
    else:
        NotImplementedError
    
    edge_index = to_undirected(edge_index)
    
    return edge_index


def cluster(emb, typ='kmeans', n_clusters=15, reorder=True, tsne_embed=True, seed=0):
    """Cluster embedding"""
    
    emb = emb.detach().numpy()
    
    clusters = dict()
    if typ=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(emb)
        clusters['n_clusters'] = n_clusters
        clusters['labels'] = kmeans.labels_
        clusters['centroids'] = kmeans.cluster_centers_
    else:
        NotImplementedError
        
    #reorder to give close clusters similar labels
    if reorder:
        pd = pairwise_distances(clusters['centroids'], metric='euclidean')
        pd += np.max(pd)*np.eye(n_clusters)
        mapping = {}
        id_old = 0
        for i in range(n_clusters):
            id_new = np.argmin(pd[id_old,:])
            while id_new in mapping.keys():
                pd[id_old,id_new] += np.max(pd)
                id_new = np.argmin(pd[id_old,:])
            mapping[id_new] = i
            id_old = id_new
            
        l = clusters['labels']
        clusters['labels'] = np.array([mapping[l[i]] for i,_ in enumerate(l)])
        clusters['centroids'] = clusters['centroids'][list(mapping.keys())]
        
    if tsne_embed:
        n_emb = emb.shape[0]
        emb = np.vstack([emb, clusters['centroids']])
        if emb.shape[1]>2:
            print('Performed t-SNE embedding on embedded results.')
            emb = TSNE(init='random',learning_rate='auto').fit_transform(emb)
            
        clusters['centroids'] = emb[n_emb:]
        emb = emb[:n_emb]       
        
        
    return emb, clusters


def parallel_proc(fun, iterable, inputs, processes=-1, desc=""):
    """Parallel processing, distribute an iterable function between processes"""
    if processes==-1:
        processes = multiprocessing.cpu_count()
    pool = Pool(processes=processes)
    fun = partial(fun, inputs)
    result = list(tqdm(pool.imap(fun, iterable), 
                            total=len(iterable), 
                            desc=desc)
                  )
    pool.close()
    pool.join()
        
    return result


def torch2np(x):
    return x.detach().to(torch.device('cpu')).numpy()


def np2torch(x):
    return torch.from_numpy(x).float()