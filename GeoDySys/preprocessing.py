#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g
from .lib import utils

def preprocessing(data, par):
    
    #compute gauges
    gauges, _ = g.compute_gauges(data, par['local_gauge'], par['n_geodesic_nb'])
    
    #compute connections
    dim_man = data.pos.shape[1]
    R = None
    if par['vector']:
        R = g.compute_connections(gauges, data.edge_index, dim_man)
        R = utils.np2torch(R)
    
    #compute kernels
    #kernels
    # kernels = g.gradient_op(data.pos, data.edge_index, gauges)
    kernels = g.DD(data.pos, data.edge_index, gauges)
    
    return gauges, R, kernels