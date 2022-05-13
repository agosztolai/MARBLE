#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import networkx as nx

from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.transforms import RandomNodeSplit

from cknn import cknneighbors_graph
from sklearn.cluster import KMeans


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
        
        G.append(G_)
        data_list.append(data_)
        
    #collate datasets
    data, slices, _ = collate(data_list[0].__class__,
                              data_list=data_list,
                              increment=True,
                              add_batch=False)
    
    split = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    split(data)
    
    return data, slices, G


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
    # pd = pairwise_distances(clusters['centroids'], metric='euclidean')
    # pd += np.max(pd)*np.eye(n_clusters)
    # new_labels = [0]
    # for i in range(n_clusters-1):
    #     ind_min = np.argmin(pd[i,:])
    #     while (ind_min==new_labels).any():
    #         pd[i,ind_min] += np.max(pd)
    #         ind_min = np.argmin(pd[i,:])
    #     new_labels.append(ind_min)
        
    # mapping = {i:new_labels[i] for i in range(n_clusters)}
    # clusters['labels'] = np.array([mapping[clusters['labels'][i]] 
    #                                 for i,_ in enumerate(clusters['labels'])])
    # clusters['centroids'] = clusters['centroids'][new_labels]
        
    return clusters