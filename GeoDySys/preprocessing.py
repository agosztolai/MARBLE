#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:13:32 2022

@author: gosztola
"""

from GeoDySys import geometry as g

def preprocessing(data, par):
    gauges, _ = g.compute_gauges(data, par['local_gauge'], par['n_geodesic_nb'])
    
    dim_man = 2
    L = g.compute_laplacian(data)
    R = g.compute_connections(gauges, L, dim_man)
    
    #kernels
    # kernel = geometry.gradient_op(data)
    kernel = g.DD(data.pos, data.edge_index, gauges)
    
    return gauges, R, kernel