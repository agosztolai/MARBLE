#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from cknn import cknneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def construct_data_object(x, y, graph_type='knn', k=10):
        
    data_list = []
    G = []
    for i, y_ in enumerate(y):
        #fit knn before adding function as node attribute
        data_ = fit_knn_graph(x[i], k=k)
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
    
    return data_train, slices, G
    
    
def split(data, test_size=0.1, val_size=0.5, seed=0):
    """
    Split training and test datasets

    Parameters
    ----------
    data : pytorch geometric data object. All entried must be torch tensors
            for this to work properly.
    test_size : float between 0 and 1, optional
        Test set as fraction of total dataset. The default is 0.1.
    val_size : float between 0 and 1, optional
        Validation set as fraction of test set. The default is 0.5.
    seed : int, optional
        Seed. The default is 0.

    Returns
    -------
    data : pytorch geometric data object.

    """
    
    n = len(data.x)
    train_id, test_id = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    test_id, val_id = train_test_split(test_id, test_size=val_size, random_state=seed)
    
    train_mask = torch.zeros(n, dtype=bool)
    test_mask = torch.zeros(n, dtype=bool)
    val_mask = torch.zeros(n, dtype=bool)
    train_mask[train_id] = True
    test_mask[test_id] = True
    val_mask[val_id] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    
    return data


def fit_knn_graph(X, k=10, graph_type='cknn'):
    
    node_feature = torch.tensor(X, dtype=torch.float)
    
    ckng = cknneighbors_graph(node_feature, n_neighbors=k, delta=1.0)    
    if graph_type=='cknn':
        edge_index = torch.tensor(list(nx.Graph(ckng).edges), dtype=torch.int64).T
    elif graph_type=='knn':
        edge_index = knn_graph(node_feature, k=k)
    else:
        NotImplementedError
    
    edge_index = to_undirected(edge_index)
    data = Data(x=node_feature, edge_index=edge_index)
    
    return data


def cluster(emb, typ='knn', n_clusters=15, seed=0):
    
    clusters = dict()
    if typ=='knn':
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(emb)
        clusters['n_clusters'] = n_clusters
        clusters['labels'] = kmeans.labels_
        clusters['centroids'] = kmeans.cluster_centers_
    else:
        NotImplementedError
        
    #reorder such that close clusters have similar label numbers
    pd = pairwise_distances(clusters['centroids'], metric='euclidean')
    new_labels = [0]
    for i in range(n_clusters-1):
        ind_min=np.array(0)
        while (ind_min==new_labels).any():
            pd[i,ind_min] += np.max(pd)
            ind_min = np.argmin(pd[i,:])
        new_labels.append(ind_min)
        
    mapping = {i:new_labels[i] for i in range(n_clusters)}
    clusters['labels'] = np.array([mapping[clusters['labels'][i]] for i,_ in enumerate(clusters['labels'])])
    clusters['centroids'] = clusters['centroids'][new_labels]
    print(new_labels)
        
    return clusters