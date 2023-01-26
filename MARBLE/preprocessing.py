#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from .lib import geometry as g
from .lib import utils

def preprocessing(data, 
                  frac_geodesic_nb=2.0, 
                  var_explained=0.9,
                  diffusion_method=None,
                  proj_man=False,
                  vector=True,
                  compute_cl=False,
                  n_workers=1):
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
    
    n, dim_emb = data.pos.shape
    dim_signal = data.x.shape[1]
    print('---- Embedding dimension: {}'.format(dim_emb))
    print('---- Signal dimension: {}\n'.format(dim_signal))
    
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
    elif dim_emb!=dim_signal:
        print('\nEmbedding dimension /= signal dimension, so manifold computations are disabled!')
    else:
        local_gauge = True
        
    #gauges
    if local_gauge:
        n_nb = int(data.degree*frac_geodesic_nb)
        try:
            gauges, Sigma, R = g.compute_tangent_bundle(data, n_geodesic_nb=n_nb)
        except:
            raise Exception('\nCould not compute gauges (possibly data is too sparse or the \
                  number of neighbours is too small)')
    else:
        gauges = torch.eye(dim_emb).repeat(n,1,1) 
            
    #Laplacian
    L = g.compute_laplacian(data)
    
    #kernels 
    kernels = g.gradient_op(data.pos, data.edge_index, gauges)
    
    #connections
    if local_gauge:
        dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
        
        print('\n---- Manifold dimension: {}'.format(dim_man))
        print('\nManifold dimension can decrease with more data. Try smaller values of stop_crit\
                 before settling on a value\n')
        
        if dim_man<dim_emb:
            kernels = [utils.tile_tensor(K, dim_emb) for K in kernels]
            kernels = [K*R for K in kernels]
            if proj_man:
                kernels = [utils.restrict_dimension(kernels[i], dim_emb, dim_man) for i in range(dim_man)]
                data.dim_man = dim_man
                
            if compute_cl:
                Lc = g.compute_connection_laplacian(data, R)
            else:
                Lc = None
                
        else:
            R, Lc = None, None
            print('\nEmbedding dimension = manifold dimension, so manifold computations are disabled!')
    else:
        R, Lc = None, None
        
    if diffusion_method == 'spectral':
        L = g.compute_eigendecomposition(L)
        Lc = g.compute_eigendecomposition(Lc)
    else:
        L, Lc = None, None
    
    data.kernels, data.L, data.Lc, data.gauges = kernels, L, Lc, gauges
        
    return data