#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from MARBLE import plotting, utils, geometry
from MARBLE.layers import Diffusion

def main():
    
    #parameters
    n = 512
    k = 0.4
    tau0 = 1
    
    x = sphere()
    y = f(x) #evaluated functions
    
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='radius', k=k)
    
    gauges, _ = geometry.compute_gauges(data, n_nb=20, processes=1)
    
    data.x = geometry.project_to_gauges(data.x, gauges)
    data.x = data.x/2
    
    R = geometry.compute_connections(gauges, data.edge_index, processes=1)
    L = geometry.compute_laplacian(data)
    Lc = geometry.compute_connection_laplacian(data, R)
    
    diffusion = Diffusion(L, Lc, tau0=tau0)
    data.x = diffusion(data.x, method='matrix_exp', normalise=True)
 
    #plot
    ax = plotting.fields(data, alpha=1)
    
    # data.x = gauges[...,0]/2
    # ax = plotting.fields(data, ax=ax, c='k')
    # data.x = gauges[...,1]/2
    # ax = plotting.fields(data, ax=ax, c='k')
    # data.x = gauges[...,2]/2
    # ax = plotting.fields(data, ax=ax, c='k')
    

def f(x):
    return np.repeat(np.array([[1,0,0]]), x.shape[0], axis=0)


def sphere():
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

if __name__ == '__main__':
    sys.exit(main())