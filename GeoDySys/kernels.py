#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize


def neighbour_vectors(x, edge_index):
    """Local edge vectors around each node"""
    
    n, dim = x.shape
    
    mask = torch.zeros([n,n],dtype=bool)
    mask[edge_index[0], edge_index[1]] = 1
    mask = mask.unsqueeze(2).repeat(1,1,dim)
    
    nvec = x.repeat(n,1,1)
    nvec = nvec - nvec.swapaxes(0,1) #Gij = xj - xi 
    nvec[~mask] = 0
    
    return nvec


def gradient_op(data):
    """Gradient operator

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


def project_gauge_to_neighbours(x, edge_index, local_gauge=None):
    """
    Project the gauge vectors to local edge vectors.
    
    Parameters
    ----------
    x : nxdim torch tensor
    edge_index : 2xE torch tensor
    local_gauge : dimxnxdim torch tensor, if None, global gauge is generated

    Returns
    -------
    F : list of nxn torch tensors of projected components
    
    """
    
    nvec = neighbour_vectors(x, edge_index) #(nxnxdim)
    
    n = x.shape[0]
    dim = x.shape[1]
    
    if local_gauge is None:
        local_gauge = torch.eye(dim)
        local_gauge = local_gauge.repeat(n,1,1)
    else:
        assert local_gauge.shape==(n,dim,dim), 'Incorrect dimensions for local_gauge.'
    
    local_gauge = local_gauge.swapaxes(0,1) #(nxdimxdim) -> (dimxnxdim)
    F = [(nvec*g).sum(-1) for g in local_gauge] #dot product in last dimension
        
    return F


def DD(data, local_gauge, order=1, include_identity=False):
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
    
    F = project_gauge_to_neighbours(data.pos, data.edge_index, local_gauge)

    if include_identity:
        K = [torch.eye(F[0].shape[0])]
    else:
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


def DA(data, local_gauge):
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
    
    F = project_gauge_to_neighbours(data, local_gauge)

    K = []
    for _F in F:
        Fhat = normalize(_F, dim=-1, p=1)
        K.append(torch.abs(Fhat))
        
    return K