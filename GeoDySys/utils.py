#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import networkx as nx

from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

from cknn import cknneighbors_graph
from sklearn.cluster import KMeans


def fit_knn_graph(X, k=10, graph_type='cknn'):
    
    ckng = cknneighbors_graph(X, n_neighbors=k, delta=1.0)
    
    if graph_type=='cknn':
        edge_index = torch.tensor(list(nx.Graph(ckng).edges), dtype=torch.int64).T
    elif graph_type=='knn':
        edge_index = knn_graph(X, k=k)
    else:
        NotImplementedError
    
    edge_index = to_undirected(edge_index)
    
    return edge_index


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