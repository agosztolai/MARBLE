# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Aug 17 14:43:41 2022

# @author: gosztola
# """

import numpy as np
import torch

import matplotlib.pyplot as plt

from GeoDySys import utils, geometry

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent


def run():
    nn = 8

    r = np.sqrt(np.arange(0.5, 5, 0.4))
    T = np.arange(0, 2*np.pi, 0.3)
    r, T = np.meshgrid(r, T)
    X = r*np.cos(T)
    Y = r*np.sin(T)
    Z = -r**2
    
    pos = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    data = utils.construct_dataset(pos, X.flatten()[:,None], graph_type='cknn', k=8)
    gauges, _ = geometry.compute_gauges(data, True, 2*nn)
    
    # L = geometry.compute_laplacian(data).todense()
    
    # X = compute_smooth_gauges(gauges, 2)
    
    #find best rotations based on least dominant coordinate
    R = geometry.compute_connections(gauges, data.edge_index, dim_man=2)
    

    # X = geometry.compute_smooth_gauges(data, gauges, p=2,
    #                                         max_iterations=100, 
    #                                         )
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pos[:,0],pos[:,1],pos[:,2], s=1)
    
    ax.quiver(pos[:,0],pos[:,1],pos[:,2], gauges[:,0,0], gauges[:,1,0], gauges[:,2,0], color='k', length = 0.5, alpha=0.5)
    ax.quiver(pos[:,0],pos[:,1],pos[:,2], gauges[:,0,1], gauges[:,1,1], gauges[:,2,1], color='k', length = 0.5, alpha=0.5)
    
    gauges = X
    ax.quiver(pos[:,0],pos[:,1],pos[:,2], gauges[:,0,0], gauges[:,1,0], gauges[:,2,0], color='r', length = 0.5, alpha=0.5)
    ax.quiver(pos[:,0],pos[:,1],pos[:,2], gauges[:,0,1], gauges[:,1,1], gauges[:,2,1], color='r', length = 0.5, alpha=0.5)

def cost_and_derivates(manifold, L, gauges):
       
    @pymanopt.function.pytorch(manifold)
    def cost(X, L=L, gauges=gauges):
                
        d = X.shape[-1]
        X = X.reshape(-1,X.shape[1]*X.shape[2])
        cost = X*L.matmul(X) 
        
        # n = X.shape[0]
        # cost = (torch.abs(L.unsqueeze(2).unsqueeze(3)*X.unsqueeze(0).repeat(n,1,1,1))**2).sum(1)
        # cost = torch.tensordot(L, X, dims=1)
        # cost = torch.tensordot(X, L, dims=1)
        # cost = torch.abs(torch.einsum('ij,jkl',L,X))
        # cost = cost.norm(dim=(1,2))
        # cost = cost.sum(axis=(1,2))**2
        # cost = - torch.einsum('ij,jkl',L,Xi).norm() - (Xi - X0).norm()
        return cost.sum()
        
    return cost, None


def compute_smooth_gauges(data, gauges, p=None, max_iterations=100, rotation_matrices=False):
    
    L = geometry.compute_laplacian(data).todense()
    L = torch.from_numpy(L)
    
    n = gauges.shape[-1]
    k = gauges.shape[0]
    if p is None:
        p = n
        
    manifold = Stiefel(n, p, k=k)
        
    cost, euclidean_gradient = cost_and_derivates(manifold, L, gauges)
    problem = pymanopt.Problem(manifold, cost)
    optimizer = SteepestDescent(max_iterations=max_iterations, verbosity=2)
    smooth_gauges = optimizer.run(problem, initial_point=np.array(gauges)).point
    
    if rotation_matrices:
        R = geometry.compute_connections(smooth_gauges, L)
        
        return smooth_gauges, R
    else:
        return smooth_gauges


# def compute_smooth_gauges(gauges, n_dom_dim):
    
#     assert len(gauges.shape)==3, 'Gauges need to be a nxdxk matrix.'
    
#     n, d, k = gauges.shape
    
#     global_frame = np.eye(d)
    
#     R = np.zeros([n,d,k])
    
#     smooth_gauges = np.zeros_like(gauges)
#     for i in range(n):
#         R = geometry.procrustes(global_frame[n_dom_dim:,:], 
#                         gauges[i,:,n_dom_dim:].T
#                                       )
#         smooth_gauges[i,:] = R@global_frame
          
#     return smooth_gauges


if __name__ == "__main__":
    run()