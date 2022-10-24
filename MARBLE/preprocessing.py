#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g
from .lib.utils import np2torch

def preprocessing(data, par):
    
    #gauges
    local_gauge = True if par['vector'] else False
    gauges, Sigma = g.compute_gauges(data, local_gauge, par['n_geodesic_nb'])
    
    #connections
    R = None
    if par['vector']:
        dim_man = g.manifold_dimension(Sigma, frac_explained=par['var_explained'])
        R = g.compute_connections(gauges, data.edge_index, dim_man)
        R = np2torch(R)
        
        print('\n---- Embedding dimension: {}'.format(data.x.shape[1]))
        print('---- Manifold dimension: {} \n'.format(dim_man))
        
        if dim_man==data.x.shape[1]:
            par['vector'] = False
    
    #kernels
    kernels = g.gradient_op(data.pos, data.edge_index, gauges)
    
    #Laplacians
    L = g.compute_laplacian(data)
    Lc = g.compute_connection_laplacian(data, R) if par['vector'] else None
    
    return R, kernels, L, Lc