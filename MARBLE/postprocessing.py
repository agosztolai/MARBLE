#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g
import numpy as np

def postprocessing(data,
                   cluster_typ='kmeans', 
                   embed_typ='umap', 
                   n_clusters=15, 
                   manifold=None,
                   seed=0):
    """
    Cluster embedding and return distance between clusters
    
    Returns
    -------
    data : PyG data object containing .emb attribute, a nx2 matrix of embedded data
    clusters : sklearn cluster object
    dist : cxc matrix of pairwise distances where c is the number of clusters
    
    """

    emb = data.emb
    
    #k-means cluster
    clusters = g.cluster(emb, cluster_typ, n_clusters, seed)
    clusters = g.relabel_by_proximity(clusters)
    clusters['slices'] = data._slice_dict['x']
    
    #compute distances between clusters
    dist, gamma, cdist = g.compute_histogram_distances(clusters)
    
    #embed into 2D via t-SNE for visualisation
    emb = np.vstack([emb, clusters['centroids']])
    emb, manifold = g.embed(emb, embed_typ, manifold)  
    emb, clusters['centroids'] = emb[:-n_clusters], emb[-n_clusters:]
    
    #store everything in data
    data.emb_2d = emb
    data.manifold = manifold
    data.clusters = clusters
    data.dist = dist
    data.gamma = gamma
    data.cdist = cdist
    
    return data
