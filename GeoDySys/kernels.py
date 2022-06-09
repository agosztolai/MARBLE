#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import scipy.sparse as scp


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
        gauge = torch.eye(data.pos.shape[1],dtype=torch.float64)
            
    elif gauge=='local':
        NotImplemented
        
    A = to_scipy_sparse_matrix(data.edge_index)
    mask = torch.tensor(A.todense(), dtype=bool)
    mask = mask[:,:,None].repeat(1,1,2)
    
    n = data.pos.shape[0]
    
    u = data.pos.repeat(n,1,1)
    u = u - torch.swapaxes(u,0,1) #uij = xj - xi 
    u[~mask] = 0
    
    F = []
    for g in gauge:
        g = g.repeat(n,n,1)
        
        _F = torch.zeros([n,n])
        ind = scp.find(A)
        for i,j in zip(ind[0], ind[1]):
            _F[i,j] = g[i,j,:].dot(u[i,j,:])
        F.append(_F)
        
    return F


def DD(data, gauge):
    """
    This function implements the directional derivative kernel 
    from Beaini et al. 2021.

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
    This function implements the directional average kernel 
    from Beaini et al. 2021.

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
        K.append(Fhat)
        
    return K