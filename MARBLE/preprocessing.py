#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g
from .lib.utils import np2torch

def preprocessing(data, par):
    
    #gauges
    print('\n---- Preprocessing ... \n')
    local_gauge = True if par['vector'] else False
    gauges, Sigma = g.compute_gauges(data, local_gauge, par['n_geodesic_nb'])
    
    #connections
    R = None
    if par['vector']:
        par['dim_man'] = g.manifold_dimension(Sigma, frac_explained=par['var_explained'])
        par['dim_signal'] = data.x.shape[1]
        R = g.compute_connections(gauges, data.edge_index, par['dim_man'])
        R = np2torch(R)
        
        if par['dim_man']==par['dim_signal']:
            par['vector'] = False
    
    #kernels
    kernels = g.gradient_op(data.pos, data.edge_index, gauges)
    
    #Laplacians
    L = g.compute_laplacian(data)
    Lc = g.compute_connection_laplacian(data, R) if par['vector'] else None
    
    return R, kernels, L, Lc, par