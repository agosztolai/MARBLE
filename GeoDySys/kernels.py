#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import scipy.sparse as scp


def project_gauge_to_neighbours(data, gauge):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    gauge : TYPE
        DESCRIPTION.

    Returns
    -------
    F : TYPE
        DESCRIPTION.

    """
    n = len(data.x)
    u = data.pos[:,None].repeat(1,n,1)
    u = torch.swapaxes(u,0,1) - u #uij = xj - xi 
    A = to_scipy_sparse_matrix(data.edge_index)
    
    mask = torch.tensor(A.todense(), dtype=bool)
    mask = mask[:,:,None].repeat(1,1,2)
    u[~mask] = 0
    
    F = []
    for g in gauge:
        g = torch.tensor(g, dtype=float)[None]
        g = g.repeat([n,1])
        g = g[:,None]
        g = g.repeat([1,n,1])
        
        _F = torch.zeros([n,n])
        ind = scp.find(A)
        for i,j in zip(ind[0], ind[1]):
            _F[i,j] = g[i,j,:].dot(u[i,j,:])
        F.append(_F)
        
    return F


def aggr_directional_derivative(data, gauge, eps=1e-8):
    
    F = project_gauge_to_neighbours(data, gauge)

    Bdx = []
    for F_ in F:
        Fhat = F_ / (torch.sum(torch.abs(F_), keepdim=True, dim=1) + eps)
        Bdx.append(Fhat - torch.diag(torch.sum(Fhat, dim=1)))
    
    return Bdx