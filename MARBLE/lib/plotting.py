#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import numpy as np
import networkx as nx
from torch_geometric.utils.convert import to_networkx

from scipy.spatial import Voronoi, voronoi_plot_2d

# =============================================================================
# Manifolds
# =============================================================================
def fields(data, 
           titles=None, 
           col=2, 
           figsize=(10,10), 
           keeplim=True, 
           ax=None, 
           save=None,
           node_size=10):
    """
    Plot scalar or vector fields

    Parameters
    ----------
    data : PyG Batch data object class created with utils.construct_dataset
    titles : list of titles
    col : int for number of columns to plot
    figsize : tuple of figure dimensions
    save : filename

    """
    if not isinstance(data, list):
        data = data.to_data_list() #split data batch 
        
    dim = data[0].pos.shape[1]
    vector = True if data[0].x.shape[1] > 1 else False
    row = int(np.ceil(len(data)/col))
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = gridspec.GridSpec(row, col, wspace=0.1, hspace=0.1, figure=fig)
    
    lims = None
    for i, d in enumerate(data):
        _, ax = create_axis(dim, grid[i], fig=fig)
        
        G = to_networkx(d, node_attrs=['pos'], edge_attrs=None, to_undirected=True,
                remove_self_loops=True)
        
        c = np.linalg.norm(d.x, axis=1) if vector else d.x.numpy()
        c, _ = set_colors(c.squeeze())
        
        graph(G,
              labels=None if vector else c,
              ax=ax,
              node_size=node_size,
              edge_width=0.5,
              edge_alpha=1.)
        
        if vector:
            pos = d.pos.numpy()
            if dim==2:
                ax.quiver(pos[:,0], pos[:,1], 
                      d.x[:,0], d.x[:,1], 
                      color=c, 
                      scale=10, 
                      scale_units='x',
                      width=0.005)
            elif dim==3:
                ax.quiver(pos[:,0], pos[:,1], pos[:,2], 
                      d.x[:,0], d.x[:,1], d.x[:,2], 
                      color=c, 
                      length=.5
                      # scale=10, 
                      # scale_units='x',
                      # width=0.005
                      )
            else:
                NotImplementedError
        
        if titles is not None:
            ax.set_title(titles[i])
            
        fig.add_subplot(ax)
        
        if lims is None:
            lims = get_limits(ax)
        set_axes(ax, lims=lims, off=True)
        
    if save is not None:
        savefig(fig, save)
        
        
def histograms(clusters, titles=None, col=2, figsize=(10,10), save=None):
    """
    Plot histograms of cluster distribution across datasets.

    Parameters
    ----------
    data : PyG Batch data object class created with utils.construct_dataset
    clusters : sklearn cluster object
    titles : list of titles
    col : int for number of columns to plot
    figsize : tuple of figure dimensions
    save : filename

    """
    
    l, s = clusters['labels'], clusters['slices']
    n_slices = len(s)-1
    l = [l[s[i]:s[i+1]]+1 for i in range(n_slices)]
    nc = clusters['n_clusters']
    
    row = int(np.ceil(n_slices/col))
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = gridspec.GridSpec(row, col, wspace=0.5, hspace=0.5, figure=fig)
    
    for i in range(n_slices):
        ax = plt.Subplot(fig, grid[i])
        
        ax.hist(l[i], 
                bins=np.arange(nc+1)+0.5, 
                rwidth=0.85, 
                density=True)
        ax.set_xticks(np.arange(nc)+1)
        ax.set_xlim([0, nc+1])
        ax.set_xlabel('Feature number')
        ax.set_ylabel('Probability density')
        
        if titles is not None:
            ax.set_title(titles[i])
            
        fig.add_subplot(ax)
        
    if save is not None:
        savefig(fig, save)
        
        
def embedding(emb, labels=None, clusters=None, titles=None, save=None):
    """
    Plot embeddings.

    Parameters
    ----------
    emb : nx2 matrix of embedded points
    labels : list of increasing integer node labels
    clusters : sklearn cluster object
    titles : list of titles
    save : filename

    """
    fig, ax = plt.subplots()
    
    assert emb.shape[0]==len(labels)
    
    if len(labels) > 1000:
        idx = np.random.choice(np.arange(len(labels)), size=1000)
        emb, labels = emb[idx], labels[idx]
     
    if labels is not None: 
        if isinstance(labels[0], int):
            colors = [f"C{i}" for i in np.arange(1, labels.max()+1)]
            cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(1, labels.max()+2), 
                                                              colors)
        elif isinstance(labels[0], float):
            labels, cbar = set_colors(labels)
            plt.colorbar(cbar)
            
    scatter = ax.scatter(emb[:,0], emb[:,1], c=labels, alpha=0.3)
    
    if clusters is not None:
        vor = Voronoi(clusters['centroids']) 
        voronoi_plot_2d(vor, ax=ax, show_vertices=False) 
        for k in range(clusters['n_clusters']):
            ax.annotate(k+1,clusters['centroids'][k,:])
    
    if titles is not None:
        handles,_ = scatter.legend_elements()
        ax.legend(handles,titles)
        
    ax.set_axis_off()
    
    if save is not None:
        savefig(fig, save)
        
        
def neighbourhoods(data,
                   clusters, 
                   hops=1,
                   norm=False, 
                   plot_graph=False,
                   save=None):
    """
    For each clustered neighbourhood type, draw one sample neighbourhood 
    from each dataset and plot.

    Parameters
    ----------
    data : PyG Batch data object class created with utils.construct_dataset
    clusters : sklearn cluster object
    hops : size of neighbourhood in number of hops
    norm : if True, then normalise values to zero mean within clusters
    plot_graph : if True, then plot the underlying graph.
    save : filename

    """
    
    vector = True if data.x.shape[1] > 1 else False
    nc = clusters['n_clusters']
    fig = plt.figure(figsize=(10, 20), constrained_layout=True)
    outer = gridspec.GridSpec(int(np.ceil(nc/3)), 3, wspace=0.2, hspace=0.2, figure=fig)
    
    data_list = data.to_data_list()
    graphs = []
    for d in data_list:
        graphs.append(to_networkx(d, 
                                  node_attrs=['pos'], 
                                  edge_attrs=None, 
                                  to_undirected=True,
                                  remove_self_loops=True))
    
    node_values = [d.x for d in data_list]
    
    for i in range(nc):
        col = 2
        row = int(np.ceil(len(data_list)/col))
        inner = gridspec.GridSpecFromSubplotSpec(row, 
                                                 col,
                                                 subplot_spec=outer[i], 
                                                 wspace=0.5, 
                                                 hspace=0.5)

        ax = plt.Subplot(fig, outer[i])
        ax.set_title("Type {}".format(i+1))
        ax.axis('off')
        fig.add_subplot(ax)
        
        n_nodes = [0] + [nx.number_of_nodes(g) for g in graphs]
        n_nodes = np.cumsum(n_nodes)

        for j, G in enumerate(graphs):
            
            label_i = clusters['labels'][n_nodes[j]:n_nodes[j+1]]==i
            label_i = np.where(label_i)[0]
            if not list(label_i):
                continue
            else:
                random_node = np.random.choice(label_i)
            
            nv = node_values[j].numpy()
            node_ids = nx.ego_graph(G, random_node, radius=hops).nodes
            node_ids = np.sort(node_ids) #sort nodes
                
            #convert node values to colors
            c = nv
            if vector:
                c = np.linalg.norm(nv, axis=1)
                
            if not norm: #set colors based on global values
                c, _ = set_colors(c)
                c = [c[i] for i in node_ids] if isinstance(c,list) else c
                nv = nv[node_ids]
            else: #first extract subgraph, then compute normalized colors
                nv = nv[node_ids]
                nv -= nv.mean()
                c, _ = set_colors(nv.squeeze())
                  
            ax = plt.Subplot(fig, inner[j])
            
            #extract subgraph with nodes sorted
            subgraph = nx.Graph()
            subgraph.add_nodes_from(sorted(G.subgraph(node_ids).nodes(data=True)))
            subgraph.add_edges_from(G.subgraph(node_ids).edges(data=True))
            
            ax.set_aspect('equal', 'box')
            if plot_graph:
                graph(subgraph,
                      labels=None,
                      ax=ax,
                      node_size=30,
                      edge_width=0.5)
            
            x = np.array(list(nx.get_node_attributes(subgraph, name='pos').values()))
            if vector:
                ax.quiver(x[:,0], x[:,1],
                          nv[:,0], nv[:,1], 
                          color=c, 
                          scale=10, 
                          scale_units='x', 
                          width=0.02)  
            else:
                ax.scatter(x[:,0], x[:,1], c=c)
            
            ax.set_frame_on(False)
            set_axes(ax, off=True)
            fig.add_subplot(ax)
            
    if save is not None:
        savefig(fig, save)
        
        
def graph(
    G,
    labels='b',
    edge_width=1,
    edge_alpha=1.,
    node_size=20,
    layout=None,
    ax=None
):
    """Plot scalar values on graph nodes embedded in 2D or 3D."""
        
    G = nx.convert_node_labels_to_integers(G)
    pos = list(nx.get_node_attributes(G, 'pos').values())
    
    if pos == []:
        if layout=='spectral':
            pos = nx.spectral_layout(G)
        else:   
            pos = nx.spring_layout(G)
            
    dim = len(pos[0])
    assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
    if ax is None:
        _, ax = create_axis(dim)
    
    if dim == 2:
        if labels is not None:
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                node_size=node_size,
                node_color=labels,
                alpha=0.8,
                ax=ax
            )

        nx.draw_networkx_edges(
            G,
            pos=pos,
            width=edge_width,
            alpha=edge_alpha,
            ax=ax
        )
    
    elif dim == 3:
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    
        if labels is not None:
            ax.scatter(*node_xyz.T, s=node_size, c=labels, ec="w")
        
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray", alpha=edge_alpha, linewidth=edge_width)  
            
    set_axes(ax, off=True)
            
    return ax

# =============================================================================
# Time series
# =============================================================================
def time_series(T, 
                X, 
                style='o', 
                node_feature=None, 
                figsize=(10,5), 
                lw=1, 
                ms=5,
                save=None):
    """
    Plot time series.

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
            
    if not isinstance(X, list):
        X = [X]
            
    fig = plt.figure(figsize=figsize, constrained_layout=True)  
    grid = gridspec.GridSpec(len(X), 1, wspace=0.5, hspace=0, figure=fig)
    
    for sp, X_ in enumerate(X):
        
        if sp == 0:
            ax = plt.Subplot(fig, grid[sp])
        else:
            ax = plt.Subplot(fig, grid[sp], sharex=ax)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
        if sp < len(X)-1:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none') 
        
        colors = set_colors(node_feature)[0]
                
        for i in range(len(X_)-2):
            if X_[i] is None:
                continue
            
            c = colors[i] if len(colors)>1 and not isinstance(colors,str) else colors
                    
            ax.plot(T[i:i+2], X_[i:i+2], style, c=c, linewidth=lw, markersize=ms)
            
            fig.add_subplot(ax)
        
    if save is not None:
        savefig(fig, save)
        
    return ax


def trajectories(X,
                 V,
                 ax=None, 
                 style='o', 
                 node_feature=None, 
                 lw=1, 
                 ms=5, 
                 arrowhead=1, 
                 arrow_spacing=3,
                 axis=False, 
                 alpha=None):
    """
    Plot trajectory in phase space. If multiple trajectories
    are given, they are plotted with different colors.

    Parameters
    ----------
    X : np array
        Positions.
    V : np array
        Velocities.
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
            
    c = set_colors(node_feature)[0]
    if alpha is not None:
        al=np.ones(len(X))*alpha
    elif len(c)>1 and not isinstance(c, str):
        al=np.abs(node_feature)/np.max(np.abs(node_feature))
    else:
        al=1
                
    if dim==2:
        if 'o' in style:
            ax.scatter(X[:,0], X[:,1], c=c, s=ms, alpha=al)
        if '-' in style:
            ax.plot(X[:,0], X[:,1], c=c, linewidth=lw, markersize=ms, alpha=al)
        if '>' in style:
            arrow_prop_dict = dict(color=c, alpha=al, lw=lw)
            skip = (slice(None, None, arrow_spacing), slice(None))
            X, V = X[skip], V[skip]
            for j in range(X.shape[0]):
                ax.quiver(X[j,0], X[j,1], V[j,0]*0.1, V[j,1]*0.1,
                          **arrow_prop_dict)
    elif dim==3:
        if 'o' in style:
            ax.scatter(X[:,0], X[:,1], X[:,2], c=c, s=ms, alpha=al)
        if '-' in style:
            ax.plot(X[:,0], X[:,1], X[:,2], c=c, linewidth=lw, markersize=ms, alpha=al)
        if '>' in style:
            arrow_prop_dict = dict(mutation_scale=arrowhead, arrowstyle='-|>', color=c, alpha=al, lw=lw)
            skip = (slice(None, None, arrow_spacing), slice(None))
            X, V = X[skip], V[skip]
            for j in range(X.shape[0]):
                a = Arrow3D([X[j,0], X[j,0]+V[j,0]], 
                            [X[j,1], X[j,1]+V[j,1]], 
                            [X[j,2], X[j,2]+V[j,2]], 
                            **arrow_prop_dict)
                ax.add_artist(a)
                
    if not axis:
        ax = set_axes(ax, off=True)
        
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
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

# =============================================================================
# Helper functions
# =============================================================================
def create_axis(*args, fig=None):
    
    dim = args[0]
    if len(args)>1:
        args = [args[i] for i in range(1,len(args))]
    else:
        args = (1,1,1)
    
    if fig is None:
        fig = plt.figure()
        
    if dim==2:
        ax = fig.add_subplot(*args)
    elif dim==3:
        ax = fig.add_subplot(*args, projection="3d")
        
    return fig, ax


def get_limits(ax):
    lims = [ax.get_xlim(), ax.get_ylim()]
    if ax.name=="3d":
        lims.append(ax.get_zlim())
        
    return lims


def set_axes(ax, lims=None, padding=0.1, off=True):
    
    if lims is not None:
        xlim = lims[0]
        ylim = lims[1]
        pad = padding*(xlim[1] - xlim[0])
        
        ax.set_xlim([xlim[0]-pad, xlim[1]+pad])
        ax.set_ylim([ylim[0]-pad, ylim[1]+pad])
        if ax.name=="3d":
            zlim = lims[2]
            ax.set_zlim([zlim[0]-pad, zlim[1]+pad])
        
    if off:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if ax.name=="3d":
            ax.set_zticklabels([])        
    
    return ax


def set_colors(color, cmap=plt.cm.coolwarm):
    
    if color is None:
        return 'C0', None
    else:
        assert isinstance(color, (list, tuple, np.ndarray))
        
    if isinstance(color[0], (float, np.floating)):
        if (color>=0).all():
            norm = plt.cm.colors.Normalize(0, np.max(np.abs(color)))
        else:    
            norm = plt.cm.colors.Normalize(-np.max(np.abs(color)), np.max(np.abs(color)))
        
        colors = []
        for i, c in enumerate(color):
            colors.append(cmap(norm(np.array(c).flatten())))
   
    elif isinstance(color[0], int):
        colors = [f"C{i}" for i in np.arange(1, color.max()+1)]
        cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(1, color.max()+2), 
                                                              colors)
        
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            
    return colors, cbar


def savefig(fig, filename, folder='../results'):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        if not Path(folder).exists():
            os.mkdir(folder)
        fig.savefig((Path(folder) / filename), bbox_inches="tight")