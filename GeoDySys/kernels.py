#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize


def neighbour_vectors(x, edge_index):
    """Spatial gradient vectors around each node"""
    
    n, dim = x.shape
    
    mask = torch.zeros([n,n],dtype=bool)
    mask[edge_index[0], edge_index[1]] = 1
    mask = mask.unsqueeze(2).repeat(1,1,dim)
    
    nvec = x.repeat(n,1,1)
    nvec = nvec - torch.swapaxes(nvec,0,1) #Gij = xj - xi 
    nvec[~mask] = 0
    
    return nvec


def gradient_op(data):
    """
    Gradient operator

    Parameters
    ----------
    nvec : TYPE
        Neighbourhood vectors.

    Returns
    -------
    G : TYPE
        Gradient matrix.

    """
    
    nvec = neighbour_vectors(data.pos, data.edge_index)
    G = torch.zeros_like(nvec)
    for i, g_ in enumerate(nvec):
        neigh_ind = torch.where(g_[:,0]!=0)[0]
        g_ = g_[neigh_ind]
        b = torch.column_stack([-1.*torch.ones((len(neigh_ind),1)),
                                torch.eye(len(neigh_ind))])
        grad = torch.linalg.lstsq(g_, b).solution
        G[i,i,:] = grad[:,[0]].T
        G[i,neigh_ind,:] = grad[:,1:].T
            
    return [G[...,i] for i in range(G.shape[-1])]


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
        gauge = torch.eye(data.pos.shape[1])
            
    elif gauge=='local':
        NotImplemented
        
    nvec = neighbour_vectors(data.pos, data.edge_index)
    n = nvec.shape[0]
    
    F = []
    for g in gauge:
        g = g.repeat(n,n,1)
        
        _F = torch.zeros([n,n])
        ind = torch.nonzero(nvec[...,0])
        for i,j in zip(ind[:,0], ind[:,1]):
            _F[i,j] = g[i,j,:].dot(nvec[i,j,:])
        F.append(_F)
        
    return F


def DD(data, gauge, order=1):
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
        
    #derivative orders
    if order>1:
        n = len(K)
        K0 = K
        for i in range(order-1):
            Ki = [K0[j]*K0[k] for j in range(n) for k in range(n)]
            K += Ki
    
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