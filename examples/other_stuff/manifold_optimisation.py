# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Aug 17 14:43:41 2022

# @author: gosztola
# """

import numpy as np
import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent

import matplotlib.pyplot as plt

from GeoDySys import utils, geometry

# from procrustes import permutation

# import geoopt
# from geoopt.optim import RiemannianSGD


# def compute_optimal_rotation(XtY):
#     n = XtY[0].shape[0]
#     U, S, Vt = np.linalg.svd(XtY)
#     # UVt = U @ Vt
    
#     V = Vt.T

#     d = np.linalg.det(V @ U.T)
#     e = np.eye(n)
#     e[-1,-1] = d

#     return V @ e @ U.T
    

def create_cost_and_derivates(manifold, gauges, L):
       
    @pymanopt.function.pytorch(manifold)
    def cost(X, gauges=gauges, L=L):
                
        cost_total = 0
        for i, g in enumerate(gauges):
            
            cost_ = torch.abs(torch.einsum('ij,jkl',L,X))
            cost_ = cost_.sum(axis=(1,2))**2
            cost_ = cost_.sum()
            cost_total += cost_
        
        return cost_total

    return cost, None


def run():
    nn = 8

    r = np.sqrt(np.arange(0.5, 5, 0.4))
    T = np.arange(0, 2*torch.pi, 0.2)
    r, T = np.meshgrid(r, T)
    X = r*np.cos(T)
    Y = r*np.sin(T)
    Z = -r**2
    
    pos = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    data = utils.construct_dataset(pos, X.flatten()[:,None], graph_type='cknn', k=8)
    gauges, _ = geometry.compute_gauges(data, True, 2*nn)
    L = geometry.compute_laplacian(data).todense()
    L = torch.from_numpy(L)
    
    # # permutation procrustes
    # for j, g_ in enumerate(g):
    #     P = permutation(g[0],g_)
    #     g[j] = P.t@g_

    k, n = pos.shape
    manifold = Stiefel(n, n, k=k)
        
    cost, euclidean_gradient = create_cost_and_derivates(
        manifold, gauges, L
    )
    problem = pymanopt.Problem(
        manifold, cost
    )

    optimizer = SteepestDescent(max_iterations=50, verbosity=2)
    X = optimizer.run(problem, initial_point=np.array(gauges)).point
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pos[:,0],pos[:,1],pos[:,2], s=1)
    
    ax.quiver(pos[:,0],pos[:,1],pos[:,2], gauges[:,0,0], gauges[:,1,0], gauges[:,2,0], color='k', length = 0.5, alpha=0.5)
    ax.quiver(pos[:,0],pos[:,1],pos[:,2], gauges[:,0,1], gauges[:,1,1], gauges[:,2,1], color='k', length = 0.5, alpha=0.5)
    
    gauges = X
    ax.quiver(pos[:,0],pos[:,1],pos[:,2], gauges[:,0,0], gauges[:,1,0], gauges[:,2,0], color='r', length = 0.5, alpha=0.5)
    ax.quiver(pos[:,0],pos[:,1],pos[:,2], gauges[:,0,1], gauges[:,1,1], gauges[:,2,1], color='r', length = 0.5, alpha=0.5)

    
    # torch.manual_seed(1)
    # learning_rate = 0.2

    # iterates = []

    # param = geoopt.ManifoldParameter(
    #         gauges[[0]].clone(), manifold=geoopt.Stiefel(canonical=False))

    # with torch.no_grad():
    #     param.proj_()
        
    # optimizer = RiemannianSGD((param,), lr=learning_rate)
    # for i in range(10):

    #     optimizer.zero_grad()
    #     loss = loss_fun(param, gauges)
    #     loss.backward()
    #     iterates.append(param.data.clone())
    #     optimizer.step()
    #     print(loss)
    
    
    # R = []
    # for i, g in enumerate(edge_index):
    #     R.append(compute_optimal_rotation(g[:p].T.matmul(torch.eye(n)[:p])))
    #     gauges[i] = R[i].T.matmul(g)


if __name__ == "__main__":
    run()