#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import plotting
from .lib import geometry as g
import numpy as np
import matplotlib.pyplot as plt
import torch

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
    data : PyG data object containing .out attribute, a nx2 matrix of embedded data
    clusters : sklearn cluster object
    dist : cxc matrix of pairwise distances where c is the number of clusters
    
    """

    if type(data) is list:
        out = np.vstack([d.out for d in data])
    else:
        out = data.out
        
    
    #k-means cluster
    clusters = g.cluster(out, cluster_typ, n_clusters, seed)
    clusters = g.relabel_by_proximity(clusters)
    
    if type(data) is list:
        slices = [d._slice_dict['x'] for i, d  in enumerate(data)]
        for i in range(1,len(slices)):
            slices[i] = slices[i] + slices[i-1][-1]
        clusters['slices'] = torch.unique(torch.cat(slices))
    else:
        clusters['slices'] = data._slice_dict['x']
        
        if data.number_of_resamples>1:
            clusters['slices'] = clusters['slices'][::data.number_of_resamples]      
    #clusters['slices'] = data._slice_dict['x']
    


    
    #compute distances between clusters
    dist, gamma, cdist = g.compute_histogram_distances(clusters)
    
    #embed into 2D via t-SNE for visualisation
    emb = np.vstack([out, clusters['centroids']])
    emb, manifold = g.embed(emb, embed_typ, manifold)  
    emb, clusters['centroids'] = emb[:-clusters['n_clusters']], emb[-clusters['n_clusters']:]
    

    #store everything in data    
    if type(data) is list:
        data_ = data[0]
        data_.emb = out
        data_.y = torch.cat([d.y for d  in data])
    else:
        data_ = data
        
    data_.emb = emb
    data_.manifold = manifold
    data_.clusters = clusters
    data_.dist = dist
    data_.gamma = gamma
    data_.cdist = cdist
    
    return data_


def compare_attractors(data, source_target):
    
    assert all(hasattr(data, attr) for attr in ['emb', 'gamma', 'clusters', 'cdist']), \
        'It looks like postprocessing has not been run...'
    
    s, t = source_target
    slices = data._slice_dict['x']
    n_slices = len(slices)-1
    s_s = range(slices[s], slices[s+1])
    s_t = range(slices[t], slices[t+1])
    
    assert s<n_slices-2 and t<n_slices-1, 'Source and target must be < number of slices!'
    assert s!=t, 'Source and target must be different!'
    
    _, ax = plt.subplots(1, 3, figsize=(10,5))
    
    #plot embedding of all points in gray
    plotting.embedding(data.emb, ax=ax[0], alpha=0.05)
    
    #get gamma matrix for the given source-target pair
    gammadist = data.gamma[s,t,...]
    np.fill_diagonal(gammadist, 0.0)
    
    #color code source features
    c = gammadist.sum(1)
    cluster_ids = set(data.clusters['labels'][s_s])
    labels = [i for i in s_s]
    for cid in cluster_ids:
        idx = np.where(cid==data.clusters['labels'][s_s])[0]
        for i in idx:
            labels[i] = c[cid]
    
    #plot source features in red
    plotting.embedding(data.emb_2d[s_s], labels=labels, ax=ax[0], alpha=1.)
    prop_dict = dict(style='>', lw=2, arrowhead=.1, \
                           axis=False, alpha=1.)
    plotting.trajectories(data.pos[s_s], data.x[s_s], 
                          ax=ax[1], 
                          node_feature=labels,
                          **prop_dict)
    ax[1].set_title('Before')
        
    #color code target features
    c = gammadist.sum(0)
    cluster_ids = set(data.clusters['labels'][s_t])
    labels = [i for i in s_t]
    for cid in cluster_ids:
        idx = np.where(cid==data.clusters['labels'][s_t])[0]
        for i in idx:
            labels[i] = -c[cid] #negative for blue color
    
    #plot target features in blue
    plotting.embedding(data.emb_2d[s_t], labels=labels, ax=ax[0], alpha=1.)
    plotting.trajectories(data.pos[s_t], data.x[s_t], 
                          ax=ax[2],
                          node_feature=labels,
                          **prop_dict)
    ax[2].set_title('After')