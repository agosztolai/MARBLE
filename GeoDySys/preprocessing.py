#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g
from .lib import utils

def preprocessing(data, par):
    
    #compute gauges
    if par['vector']:
        local_gauge=True
    else:
        local_gauge=False
        
    gauges, _ = g.compute_gauges(data, local_gauge, par['n_geodesic_nb'])
    
    #compute connections
    R = None
    if par['vector']:
        dim_man = data.pos.shape[1]
        R = g.compute_connections(gauges, data.edge_index, dim_man)
        R = utils.np2torch(R)
    
    #compute kernels
    #kernels
    # kernels = g.gradient_op(data.pos, data.edge_index, gauges)
    kernels = g.DD(data.pos, data.edge_index, gauges)
    
    return gauges, R, kernels