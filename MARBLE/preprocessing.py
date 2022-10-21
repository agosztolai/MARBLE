#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g
from .lib import utils

def preprocessing(data, par):
    
    #gauges
    local_gauge = True if par['vector'] else False
    gauges, Sigma = g.compute_gauges(data, local_gauge, par['n_geodesic_nb'])
    
    #connections
    R = None
    if par['vector']:
        dim_man = g.manifold_dimension(Sigma, fraction_exp=0.9)
        
        # print('\n---- Embedding dimension: {}, manifold dimension: {} \n'\
        #       .format(data.shape[1], dim_man))
        
        R = g.compute_connections(gauges, data.edge_index, dim_man)
        R = utils.np2torch(R)
    
    #kernels
    # kernels = g.gradient_op(data.pos, data.edge_index, gauges)
    kernels = g.DD(data.pos, data.edge_index, gauges)
    
    #Laplacians
    L = g.compute_laplacian(data)
    Lc = g.compute_connection_laplacian(data, R) if par['vector'] else None
    
    return R, kernels, L, Lc