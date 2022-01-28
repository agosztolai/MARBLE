#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path
import os
import networkx as nx


def time_series(T,X, ax=None, style='o', node_feature=None, lw=1, ms=5):
    """
    Plot time series coloured by curvature.

    Parameters
    ----------
    X : np array or list[np array]
        Trajectories.
    style : string
        Plotting style. The default is 'o'.
    color: bool
        Color lines. The default is True.
    lw : int
        Line width.
    ms : int
        Marker size.

    Returns
    -------
    ax : matplotlib axes object.

    """
            
    if ax is None:
        _, ax = create_axis(2)
    
    colors = set_colors(node_feature)
            
    for i in range(len(X)-2):
        if X[i] is None:
            continue
        
        c = colors[i] if len(colors)>1 else colors
                
        ax.plot(T[i:i+2], X[i:i+2], style, c=c, linewidth=lw, markersize=ms)
        
    return ax


def trajectories(X, ax=None, style='o', node_feature=None, lw=1, ms=5, axis=False, alpha=None):
    """
    Plot trajectory in phase space. If multiple trajectories
    are given, they are plotted with different colors.

    Parameters
    ----------
    X : np array
        Trajectories.
    style : string
        Plotting style. The default is 'o'.
    node_feature: bool
        Color lines. The default is None.
    lw : int
        Line width.
    ms : int
        Marker size.

    Returns
    -------
    ax : matplotlib axes object.

    """
            
    dim = X.shape[1]
    assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
    if ax is None:
        _, ax = create_axis(dim)
            
    color = set_colors(node_feature)
    if alpha is not None:
        al=np.ones(len(X))*alpha
    elif len(color)>1:
        al=np.abs(node_feature)/np.max(np.abs(node_feature))
    else:
        al=1
                
    if dim==2:
        ax.scatter(X[:, 0], X[:, 1], c=color, s=ms, alpha=al)
        # ax.plot(X_l[:, 0], X_l[:, 1], style, c=c, linewidth=lw, markersize=ms, alpha=al)
        if style=='-':               
            for j in range(X.shape[0]):
                if (j+1)%2==0 and j>0:
                    a = ax.arrow(X[j,0], X[j,1], X[j,0]-X[j-1,0], X[j,1]-X[j-2,1])
                    ax.add_artist(a)
    elif dim==3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=ms, alpha=al)
        # ax.plot(X_l[:, 0], X_l[:, 1], X_l[:, 2], style, c=c, linewidth=lw, markersize=ms,alpha=al)
        if style=='-':
            for j in range(X.shape[0]):
                if (j+1)%2==0 and j>0:
                    a = Arrow3D([X[j-1,0], X[j,0]], [X[j-1,1], X[j,1]], 
                                [X[j-1,2], X[j,2]], mutation_scale=ms, 
                                 lw=lw, arrowstyle="-|>", color=c, alpha=al[j])
                    ax.add_artist(a)
                
    if not axis:
        ax = set_axes(ax, data=None, off=True)
        
    return ax


def transition_diagram(centers, P, ax=None, radius=None, lw=1, ms=1, alpha=0.3, exclude_zeros=False):
    
    dim = centers.shape[1]
    assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
    if ax is None:
        _, ax = create_axis(dim)
        
    colors = set_colors(P)
    colors = np.array(colors)
    
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if exclude_zeros and P[i,j]==0:
                continue
            if radius is not None:
                dist = np.max(np.abs(centers[i]-centers[j]))
                if radius < dist or np.sum(dist)==0:
                    continue
            a = Arrow3D([centers[i][0], centers[j][0]], [centers[i][1], centers[j][1]], 
                        [centers[i][2], centers[j][2]], mutation_scale=ms, 
                        lw=lw, arrowstyle="-|>", color=colors[i,j], alpha=alpha)
            ax.add_artist(a)
    
    return ax


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


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
        fig, ax = create_axis(2)

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
    
    _savefig(fig, folder, filename, ext=ext)
    
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
        _, ax = create_axis(dim)
    
    pc = plotCubeAt2(centers,sizes,colors=None, edgecolor="k", linewidths=0.2, alpha=alpha)
    ax.add_collection3d(pc)
    
    ax = set_axes(ax, data=centers, off=True)
        
    return ax


def create_axis(dim):
    
    fig = plt.figure()
    if dim==2:
        ax = plt.axes()
    if dim==3:
        ax = plt.axes(projection="3d")
        
    return fig, ax


def set_axes(ax,data=None, padding=0.1, off=True):
    
    if data is not None:
        cmin = data.min(0)
        cmax = data.max(0)
        pad = padding*(cmax - cmin)
        
        ax.set_xlim([cmin[0]-pad[0],cmax[0]+pad[0]])
        ax.set_ylim([cmin[1]-pad[1],cmax[1]+pad[1]])
        if ax.name=="3d":
            ax.set_zlim([cmin[2]-pad[2],cmax[2]+pad[2]])
        
    if off:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if ax.name=="3d":
            ax.set_zticklabels([])        
    
    return ax


def set_colors(color):
    
    if color is None:
        colors = ['C0']
    else:
        if isinstance(color, (list, tuple, np.ndarray)):
            cmap = plt.cm.coolwarm
            if (color>=0).all():
                norm = plt.cm.colors.Normalize(0, np.max(np.abs(color)))
            else:    
                norm = plt.cm.colors.Normalize(-np.max(np.abs(color)), np.max(np.abs(color)))
            cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            plt.colorbar(cbar)
            colors = []
            for i, c in enumerate(color):
                colors.append(cmap(norm(np.array(c).flatten())))
        else:
            colors = [color]
        
    return colors


def _savefig(fig, folder, filename, ext):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        if not Path(folder).exists():
            os.mkdir(folder)
        fig.savefig((Path(folder) / filename).with_suffix(ext), bbox_inches="tight")