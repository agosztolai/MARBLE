#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .lib import geometry as g

def preprocessing(data, par):
    
    #compute gauges
    gauges, _ = g.compute_gauges(data, par['local_gauge'], par['n_geodesic_nb'])
    
    #compute connections
    dim_man = 2
    if par['vector']:
        R = g.compute_connections(gauges, data.edge_index, dim_man)
    else:
        R = None
    
    #compute kernels
    #kernels
    # kernel = geometry.gradient_op(data)
    kernel = g.DD(data.pos, data.edge_index, gauges)
    
    return gauges, R, kernel