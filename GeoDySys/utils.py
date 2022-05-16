#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import networkx as nx
import numpy as np

from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

from cknn import cknneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE


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


def cluster(emb, typ='knn', n_clusters=15, reorder=True, tsne_embed=True, seed=0):
    
    emb = emb.detach().numpy()
    
    clusters = dict()
    if typ=='knn':
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(emb)
        clusters['n_clusters'] = n_clusters
        clusters['labels'] = kmeans.labels_
        clusters['centroids'] = kmeans.cluster_centers_
    else:
        NotImplementedError
        
    #reorder such that close clusters have similar label numbers
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