#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys
# from math import comb
import random
from GeoDySys import utils, plotting
from GeoDySys.lib.solvers import simulate_ODE
random.seed(a=0)
import discretisation as disc

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

            
def main():
    
    """1. Simulate system"""
    # x0 = [0.1,0.1] 
    # par = {'mu': 0.1}
    # fun = 'vanderpol'
    
    par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
    fun = 'lorenz'
    
    # par['sigma']*((par['sigma']+par['beta']+3)/(par['sigma']-par['beta']-1))
    
    n=1000
    x0 = [-8.0, 7.0, 27.0]
    t = np.linspace(0, 50, n)
    t_ind = np.arange(n)
    mu, sigma = 0, 1 # mean and standard deviation
    X = simulate_ODE(fun, t, x0, par)#, mu=mu, sigma=sigma)
    t_sample = t_ind
    
    
    """2. Random project and then delay embed"""
    # from GeoDySys import time_series
    # x = time_series.random_projection(X, seed=0)
    # dim = 3
    # X = time_series.delay_embed(X[:,0],dim,tau=-1)
    # t_sample = t_sample[:-dim]
    
    
    """3. Discretisation"""
    centers, sizes, labels = disc.kmeans_part(X, 100)
    # centers, sizes, labels = discretisation.maxent_part(X, dim, .1)
    
    """4. Markov transition operator"""
    # P = op_calculations.get_transition_matrix(t_ind,labels,T=5)
     
    """5. Curvature matrix"""
    # K = curvature.get_curvature_matrix(X,t_ind,labels,T=10,Tmax=50)
    
    # K = np.clip(K, None, 0.1)
    
    """Plotting"""
    ax = plotting.trajectories(X, node_feature=None, style='-', lw=.5, ms=1,alpha=.5)
    
    #discretisation
    # ax = discretisation(centers, sizes, ax=None, alpha=0.0)
    # ax = transition_diagram(centers, K, ax=ax, radius=5, lw=2, ms=1, alpha=1, exclude_zeros=True)
    plt.savefig('../../results/discretisation.svg')
    
    #transition matrix
    # plt.figure()
    # plt.imshow(P.todense())
    
    # plt.figure()
    # plt.imshow(K)
    #
    #curvature across time horizons
    # plot.plot_curvatures(times,kappas,ylog=True)
    
    
    # def transition_diagram(centers, P, ax=None, radius=None, lw=1, ms=1, alpha=0.3, exclude_zeros=False):
        
    #     dim = centers.shape[1]
    #     assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
        
    #     if ax is None:
    #         _, ax = plotting.create_axis(dim)
            
    #     colors = plotting.set_colors(P)
    #     colors = np.array(colors)
        
    #     for i in range(P.shape[0]):
    #         for j in range(P.shape[0]):
    #             if exclude_zeros and P[i,j]==0:
    #                 continue
    #             if radius is not None:
    #                 dist = np.max(np.abs(centers[i]-centers[j]))
    #                 if radius < dist or np.sum(dist)==0:
    #                     continue
    #             a = Arrow3D([centers[i][0], centers[j][0]], [centers[i][1], centers[j][1]], 
    #                         [centers[i][2], centers[j][2]], mutation_scale=ms, 
    #                         lw=lw, arrowstyle="-|>", color=colors[i,j], alpha=alpha)
    #             ax.add_artist(a)
        
    #     return ax
        

def plot_curvatures(
    times,
    kappas,
    ylog=True,
    folder="figures",
    filename="curvature",
    ext=".svg",
    ax=None
):
    """Plot edge curvature."""
    if ax is None:
        fig, ax = plotting.create_axis(2)

    for kappa in kappas.T:
        ax.plot(times, kappa, c='k', lw=0.5, alpha=0.1)

    if ylog:
        ax.set_xscale("symlog")
        
    ax.axhline(0, ls="--", c="k")
    # ax.axis([np.log10(times[0]), np.log10(times[-1]), np.min(kappas), 1])
    ax.set_xlabel(r"Time horizon, $log_{10}(T)$")
    if ylog:
        ax.set_ylabel(r"Curvature, $log_{10}\kappa_t(T)$")
    else:
        ax.set_ylabel(r"Curvature, $\kappa_t(T)$")
    
    utils._savefig(fig, folder, filename, ext=ext)
    
    return fig, ax
    
    
def cuboid_data2(o, size=(1,1,1)):
    
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
          [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
          [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
          [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
          [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
          [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    
    return X


def plotCubeAt2(centers,sizes=None,colors=None, **kwargs):
    
    if not isinstance(colors,(list,np.ndarray)):
        colors=["C7"]*len(centers)
    if not isinstance(sizes,(list,np.ndarray)):
        sizes=[(1,1,1)]*len(centers)
        
    for i in range(centers.shape[0]):
        centers[i]-=sizes[i]/2
    
    g = []
    for p,s,c in zip(centers,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
        
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)


def discretisation(centers, sizes, ax=None, alpha=0.2):
    """
    Plot the tesselation of the state space as a set of boxes.
    """
        
    dim = centers.shape[1]
    assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
    if ax is None:
        _, ax = plotting.create_axis(dim)
    
    pc = plotCubeAt2(centers,sizes,colors=None, edgecolor="k", linewidths=0.2, alpha=alpha)
    ax.add_collection3d(pc)
    
    ax = plotting.set_axes(ax, data=centers, off=True)
        
    return ax

if __name__ == '__main__':
    sys.exit(main())
