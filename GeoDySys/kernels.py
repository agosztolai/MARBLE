#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import scipy.sparse as scp


def project_gauge_to_neighbours(data, gauge):
    """
    This function projects the gauge vectors into a local non-orthonormal
    unit vectors defined by the edges pointing outwards from a given node.
    
    Parameters
    ----------
    data : pytorch geometric data object containing .pos and .edge_index
    gauge : list
        Unit vectors of Euclidean coordinate system.

    Returns
    -------
    F : torch tensor
        Each row contains vectors whose components are projections (inner products)
        of the gauge to edge vectors.

    """
    n = data.pos.shape[0]
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
    """
    This function implements the directional derivative kernel 
    from Beaini et al. 2021.

    Parameters
    ----------
    data : pytorch geometric data object containing .pos and .edge_index
    gauge : list
        Unit vectors of Euclidean coordinate system.
    eps : float, optional
        Small value to avoid numerical blow-ups. The default is 1e-8.

    Returns
    -------
    Bdx : list[torch tensor]
        Anisotropic kernel.

    """
    
    F = project_gauge_to_neighbours(data, gauge)

    Bdx = []
    for F_ in F:
        Fhat = F_ / (torch.sum(torch.abs(F_), keepdim=True, dim=1) + eps)
        Bdx.append(Fhat - torch.diag(torch.sum(Fhat, dim=1)))
    
    return Bdx