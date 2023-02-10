#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import sys
from MARBLE import plotting, utils, geometry
from MARBLE.layers import Diffusion
import matplotlib.pyplot as plt

def main():
    
    #parameters
    k = 0.4
    tau0 = 1
    
    x = sphere()
    y = f(x) #evaluated functions
    
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='radius', k=k, n_geodesic_nb=10, proj_man=True, compute_cl=True)
    
    #test the connection computation
    data.x = geometry.map_to_local_gauges(data.x, data.gauges, 3) 
    gauges, Sigma, R = geometry.compute_tangent_bundle(data, n_geodesic_nb=10)
    
    gauges, L, Lc = data.gauges, data.L, data.Lc
    
    data.x = data.x/2
    
    # diffusion = Diffusion(tau0=tau0)
    # data.x = diffusion(data.x, L, Lc=Lc, method='matrix_exp', normalise=True)
    
    ind = np.arange(220).reshape(20,11)[:,5]
 
    #plot
    ax = plotting.fields(data, alpha=1)
    ax[0].plot(x[:,0].reshape(20,11)[:,5],
               x[:,1].reshape(20,11)[:,5],
               x[:,2].reshape(20,11)[:,5])
    
    data.x = gauges[...,0]/2
    plotting.fields(data, color='k')
    # plt.savefig('gauge1.svg')
    
    data.x = gauges[...,1]/2
    plotting.fields(data, color='k')
    # plt.savefig('gauge2.svg')
    
    data.x = gauges[...,2]/2
    plotting.fields(data, color='k')
    # plt.savefig('gauge3.svg')
    
    vectors_on_meridian = torch.zeros_like(gauges[...,2])
    vectors_on_meridian[ind] = gauges[...,2][ind]
    data.x = vectors_on_meridian
    plotting.fields(data, color='k')
    

def f(x):
    return np.repeat(np.array([[1,0,0]]), x.shape[0], axis=0)


def sphere():
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:11j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

if __name__ == '__main__':
    sys.exit(main())