#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from pathlib import Path
import os
import networkx as nx


def trajectories(X, ax=None, style='o', color=None, dim=3, lw=1, ms=5):
    """
    Plot trajectory in phase space in dim dimensions. If multiple trajectories
    are given, they are plotted with different colors.

    Parameters
    ----------
    X : np array or list[np array]
        Trajectories.
    style : string
        Plotting style. The default is 'o'.
    color: bool
        Color lines. The default is True.
    dim : int, optionel
        Dimension of the plot. The default is 3.
    lw : int
        Line width.
    ms : int
        Marker size.

    Returns
    -------
    ax : matplotlib axes object.

    """
    
    if not isinstance(X, list):
        X = [X]
        
    assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
    if ax is None:
        fig = plt.figure()
        if dim==2:
            ax = plt.axes()
        if dim==3:
            ax = plt.axes(projection="3d")
    
    if color is None:
        if len(X)>1:
            colors = plt.cm.jet(np.linspace(0, 1, len(X)))
        else:
            c = 'C0'
    else:
        if isinstance(color, (list, tuple, np.ndarray)):
            cmap = plt.cm.coolwarm
            norm = plt.cm.colors.Normalize(-max(abs(color)), max(abs(color)))
            cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            plt.colorbar(cbar)
        else:
            c = color
            
    for i,X_l in enumerate(X):
        if X_l is None:
            continue
        
        if len(X)>1 and color is None:
            c=colors[i]
        if isinstance(color, (list, tuple, np.ndarray)):
            if ~np.isnan(color[i]):
                c=cmap(norm(color[i]))
            else:
                continue
                
        if dim==2:
            ax.plot(X_l[:, 0], X_l[:, 1], style, c=c, linewidth=lw, markersize=ms)
            if style=='-':               
                for j in range(X_l.shape[0]):
                    if (j+1)%2==0 and j>0:
                        a = ax.arrow(X_l[j,0], X_l[j,1], X_l[j,0]-X_l[j-1,0], X_l[j,1]-X_l[j-2,1])
                        ax.add_artist(a)
                ax.scatter(X_l[0, 0], X_l[0, 1], color=c, s=ms, facecolors='none')
                ax.scatter(X_l[-1, 0], X_l[-1, 1], color=c, s=ms)
        if dim==3:
            ax.plot(X_l[:, 0], X_l[:, 1], X_l[:, 2], style, c=c, linewidth=lw, markersize=ms)
            if style=='-':
                for j in range(X_l.shape[0]):
                    if (j+1)%2==0 and j>0:
                        a = Arrow3D([X_l[j-1,0], X_l[j,0]], [X_l[j-1,1], X_l[j,1]], 
                                    [X_l[j-1,2], X_l[j,2]], mutation_scale=ms, 
                                    lw=lw, arrowstyle="-|>", color=c)
                        ax.add_artist(a)
                # ax.scatter(X_l[0, 0], X_l[0, 1], X_l[0, 2], color=c, s=ms, facecolors='none')
                # ax.scatter(X_l[-1, 0], X_l[-1, 1], X_l[-1, 2], color=c, s=ms)
        
    return ax


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_curvatures(
    times,
    kappas,
    ylog=True,
    folder="figures",
    filename="curvature",
    ext=".svg",
    ax=None,
    figsize=(5, 4),
):
    """Plot edge curvature."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None

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


def plot_graph(
    graph,
    edge_width=1,
    node_colors=None,
    node_size=20,
    show_colorbar=True,
    ax=None
):
    """Plot the curvature on the graph."""
    
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        
    pos = list(nx.get_node_attributes(graph, "pos").values())
    if pos == []:
        pos = nx.spring_layout(graph)

    if node_colors is not None:
        cmap = plt.cm.coolwarm
        vmin = min(node_colors)
        vmax = max(node_colors)
    else:
        cmap, vmin, vmax = None, None, None

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_size=node_size,
        node_color=node_colors,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        ax=ax
    )

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        width=edge_width,
        # edge_color=edge_color,
        # edge_cmap=cmap,
        # edge_vmin=vmin,
        # edge_vmax=vmax,
        alpha=0.5,
        ax=ax
    )

    if show_colorbar:
        norm = plt.cm.colors.Normalize(vmin, vmax)
        edges = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(edges)

    plt.axis("off")


def _savefig(fig, folder, filename, ext):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        if not Path(folder).exists():
            os.mkdir(folder)
        fig.savefig((Path(folder) / filename).with_suffix(ext), bbox_inches="tight")