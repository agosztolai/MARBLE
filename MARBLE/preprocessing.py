#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g
import torch

def preprocessing(data, par):
    """
    Compute geometric objects used later: local gauges, Levi-Civita connections
    gradient kernels, scalar and connection laplacians.

    Parameters
    ----------
    data : pytorch geometric data object
    par : dictionary of parameters

    Returns
    -------
    R : (nxnxdxd) tensor of L-C connectionc (dxd) matrices
    kernels : list of d (nxn) matrices of directional kernels
    L : (nxn) matrix of scalar laplacian
    Lc : (ndxnd) matrix of connection laplacian
    par : updated dictionary of parameters

    """
        
    #disable vector computations if 1) signal is scalar or 2) embedding dimension
    #is <= 2. In case 2), either M=R^2 (manifold is whole space) or case 1).
    if par['vector'] and par['dim_emb']>2:
        local_gauge = True
    else:
        local_gauge = False
        par['vector'] = False
    
    #gauges
    gauges, Sigma = g.compute_gauges(data, local_gauge, par['n_geodesic_nb'])
    
    #kernels
    kernels = g.gradient_op(data.pos, data.edge_index, gauges)
    
    #Laplacian
    L = g.compute_laplacian(data)
    if par['diffusion_method'] == 'spectral':
        L = g.compute_eigendecomposition(L)
    
    #connections
    if par['vector']:
        par['dim_man'] = g.manifold_dimension(Sigma, frac_explained=par['var_explained'])
        
        if par['dim_man']==par['dim_emb']:
            par['vector'] = False
        else:
            R = g.compute_connections(gauges, data.edge_index, par['dim_man'])
            Lc = g.compute_connection_laplacian(data, R)
            if par['diffusion_method'] == 'spectral':
                Lc = g.compute_eigendecomposition(Lc)
    else:
        R = None
        Lc = None
        
    #move to gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if isinstance(L, tuple):
        L = [L_.to(device) for L_ in L]
    else:
        L = L.to(device)
    if isinstance(Lc, tuple):
        Lc = [Lc_.to(device) for Lc_ in Lc]
    else:
        Lc = Lc.to(device) if Lc is not None else None
    R = R.to(device) if R is not None else None
    kernels = [K.to(device) for K in kernels]
    
    return R, kernels, L, Lc, par