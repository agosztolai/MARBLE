#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize


def neighbor_vectors(data):
    """Spatial gradient vectors around each node"""
    
    x = data.pos
    n, dim = x.shape
    
    mask = torch.zeros([n,n],dtype=bool)
    mask[data.edge_index[0],data.edge_index[1]] = 1
    mask = mask.unsqueeze(2).repeat(1,1,dim)
    
    G = x.repeat(n,1,1)
    G = G - torch.swapaxes(G,0,1) #Gij = xj - xi 
    G[~mask] = 0
    
    return G


def gradient_op(G_diff):
    
    G_ls = torch.zeros_like(G_diff, dtype=torch.float64)
    for i, g_ in enumerate(G_diff):
        neigh_ind = torch.where(g_[:,0]!=0)[0]
        g_ = g_[neigh_ind]
        b = torch.column_stack([-1.*torch.ones((len(neigh_ind),1),dtype=torch.float64),
                                torch.eye(len(neigh_ind),dtype=torch.float64)])
        grad = torch.linalg.lstsq(g_, b).solution
        G_ls[i,i,:] = grad[:,[0]].T
        G_ls[i,neigh_ind,:] = grad[:,1:].T
            
    return G_ls


def project_gauge_to_neighbours(data, gauge='global'):
    """
    Project the gauge vectors into a local non-orthonormal
    unit vectors defined by the edges pointing outwards from a given node.
    
    Parameters
    ----------
    data : pytorch geometric data object containing .pos and .edge_index
    gauge : 'global' or 'local'

    Returns
    -------
    F : torch tensor
        Each row contains vectors whose components are projections (inner products)
        of the gauge to edge vectors.

    """
    
    if gauge=='global':
        gauge = torch.eye(data.pos.shape[1], dtype=torch.float64)
            
    elif gauge=='local':
        NotImplemented
        
    G = neighbor_vectors(data)
    n = G.shape[0]
    
    F = []
    for g in gauge:
        g = g.repeat(n,n,1)
        
        _F = torch.zeros([n,n])
        ind = torch.nonzero(G[...,0])
        for i,j in zip(ind[:,0], ind[:,1]):
            _F[i,j] = g[i,j,:].dot(G[i,j,:])
        F.append(_F)
        
    return F


def DD(data, gauge):
    """
    Directional derivative kernel from Beaini et al. 2021.

    Parameters
    ----------
    data : pytorch geometric data object containing .pos and .edge_index
    gauge : list
        Unit vectors of Euclidean coordinate system.

    Returns
    -------
    K : list[torch tensor]
        Anisotropic kernel.

    """
    
    F = project_gauge_to_neighbours(data, gauge)

    K = []
    for _F in F:
        Fhat = normalize(_F, dim=-1, p=1)
        K.append(Fhat - torch.diag(torch.sum(Fhat, dim=1)))
    
    return K


def DA(data, gauge):
    """
    Directional average kernel from Beaini et al. 2021.

    Parameters
    ----------
    data : pytorch geometric data object containing .pos and .edge_index
    gauge : list
        Unit vectors of Euclidean coordinate system.

    Returns
    -------
    K : list[torch tensor]
        Anisotropic kernel.

    """
    
    F = project_gauge_to_neighbours(data, gauge)

    K = []
    for _F in F:
        Fhat = normalize(_F, dim=-1, p=1)
        K.append(torch.abs(Fhat))
        
    return K