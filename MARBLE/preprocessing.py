#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g

def preprocessing(data, 
                  frac_geodesic_nb=2.0, 
                  var_explained=0.9, 
                  diffusion_method='spectral'):
    """
    Compute geometric objects used later: local gauges, Levi-Civita connections
    gradient kernels, scalar and connection laplacians.

    Parameters
    ----------
    data : pytorch geometric data object
    frac_geodesic_nb: fraction of geodesic neighbours (relative to node degree) 
    to map to tangent space
    var_explained: fraction of variance explained by the local gauges
    diffusion: 'spectral' or 'matrix_exp'

    Returns
    -------
    R : (nxnxdxd) tensor of L-C connectionc (dxd) matrices
    kernels : list of d (nxn) matrices of directional kernels
    L : (nxn) matrix of scalar laplacian
    Lc : (ndxnd) matrix of connection laplacian
    par : updated dictionary of parameters

    """
    
    dim_emb = data.pos.shape[1]
    dim_signal = data.x.shape[1]
    print('---- Embedding dimension: {}'.format(dim_emb))
    print('---- Signal dimension: {}'.format(dim_signal))
    
    #disable vector computations if 1) signal is scalar or 2) embedding dimension
    #is <= 2. In case 2), either M=R^2 (manifold is whole space) or case 1).
    if dim_signal==1:
        print('\n Signal dimension is 1, so manifold computations are disabled!')
        local_gauge = False
    elif dim_emb<=2:
        print('\n Embedding dimension <= 2, so manifold computations are disabled!')
        local_gauge = False
    else:
        local_gauge = True
    
    #gauges
    n_geodesic_nb = int(data.degree*frac_geodesic_nb)
    gauges, Sigma = g.compute_gauges(data, local_gauge, n_geodesic_nb)
    
    #Laplacian
    L = g.compute_laplacian(data)        
    
    #connections
    if local_gauge:
        dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
        
        print('---- Manifold dimension: {}'.format(dim_man))
        
        if dim_man<dim_emb:
            R = g.compute_connections(gauges, data.edge_index, dim_man)
            Lc = g.compute_connection_laplacian(data, R)
        else:
            print('\n Embedding dimension = manifold dimension, so \
                      manifold computations are disabled!')
    else:
        R = None
        Lc = None
        
    if diffusion_method == 'spectral':
        L = g.compute_eigendecomposition(L)
        Lc = g.compute_eigendecomposition(Lc)
        
    #kernels
    kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        
    data.R, data.kernels, data.L, data.Lc = R, kernels, L, Lc
        
    return data