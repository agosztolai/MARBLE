#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g

def preprocessing(data, 
                  frac_geodesic_nb=2.0, 
                  var_explained=0.9, 
                  diffusion_method='spectral', 
                  vector=True):
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
    if not vector:
        local_gauge=False
        print('\nVector computations are disabled')
    elif dim_signal==1:
        print('\nSignal dimension is 1, so manifold computations are disabled!')
        local_gauge = False
    elif dim_emb<=2:
        print('\nEmbedding dimension <= 2, so manifold computations are disabled!')
        local_gauge = False
    else:
        local_gauge = True
        
    #gauges
    n_nb = int(data.degree*frac_geodesic_nb)
    try:
        gauges, Sigma = g.compute_gauges(data, local_gauge, n_nb)
    except:
        local_gauge = False
        gauges, Sigma = g.compute_gauges(data, local_gauge, n_nb)
        print('Could not compute gauges (possibly data is too sparse or the \
              number of neighbours is too small) Manifold computations are disabled!')
    
# =============================================================================
#     Debug
# =============================================================================
    
    import numpy as np
    import torch
    for i, ga in enumerate(gauges):
        t = np.random.uniform(low=0,high=2*np.pi)
        R = np.array([[np.cos(t), -np.sin(t)], 
                            [np.sin(t),  np.cos(t)]])
        gauges[i] = torch.tensor(R, dtype=torch.float32)@ga
        
    print(gauges)
        
    #Laplacian
    L = g.compute_laplacian(data)
    
    #connections
    if local_gauge:
        dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
        # dim_man=2
        
        print('\n---- Manifold dimension: {}'.format(dim_man))
        print('\nManifold dimension can decrease with more data. Try smaller values of stop_crit\
                 before settling on a value')
        
        if dim_man<dim_emb:
            R = g.compute_connections(gauges, data.edge_index, dim_man)
            Lc = g.compute_connection_laplacian(data, R)
        else:
            R, Lc = None, None
            print('\nEmbedding dimension = manifold dimension, so manifold computations are disabled!')
                
    else:
        R, Lc = None, None
        
    if diffusion_method == 'spectral':
        L = g.compute_eigendecomposition(L)
        Lc = g.compute_eigendecomposition(Lc)
        
    #kernels
    kernels = g.gradient_op(data.pos, data.edge_index, gauges)
    
    data.R, data.gauges, data.kernels, data.L, data.Lc = R, gauges, kernels, L, Lc
        
    return data