#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
from MARBLE import plotting, utils, geometry
from MARBLE.layers import Diffusion

def main():
    
    #parameters
    n = 512
    k = 30
    tau0 = 100
    
    # f1: constant, f2: linear, f3: parabola, f4: saddle
    x = geometry.sample_2d(n, [[-1,-1],[1,1]], 'random')
    y = f(x) #evaluated functions
    
    # #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
    
    gauges, _ = geometry.compute_gauges(data, False, 50, processes=1)
    R = geometry.compute_connections(gauges, data.edge_index, processes=1)
    L = geometry.compute_laplacian(data)
    Lc = geometry.compute_connection_laplacian(data, R)
    
    diffusion = Diffusion(L, Lc, tau0=tau0)
    data.x = diffusion(data.x, method='matrix_exp')
 
    #plot
    plotting.fields(data)
    
   # plot_functions(data, titles=titles, gauges=gauges) #sampled functions


def f(x):
    eps = 1e-1
    norm = np.sqrt((x[:,[0]]-1)**2 + x[:,[1]]**2 + eps)
    u = x[:,[1]]/norm
    v = -(x[:,[0]]-1)/norm
    return np.hstack([u,v])


def plot_gauges(data, ax, gauges):
    x = data.pos
    ax.quiver(x[:,0],x[:,1],gauges[:,0,0], gauges[:,0,1], scale=20, color='k',width=0.002, headwidth=2)
    ax.quiver(x[:,0],x[:,1],gauges[:,1,0], gauges[:,1,1], scale=20, color='k',width=0.002, headwidth=2)


if __name__ == '__main__':
    sys.exit(main())