#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path
import os
import networkx as nx
import matplotlib.gridspec as gridspec
from torch_geometric.utils.convert import to_networkx

from scipy.spatial import Voronoi, voronoi_plot_2d 



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
        
        c = colors[i] if len(colors)>1 and not isinstance(colors,str) else colors
                
        ax.plot(T[i:i+2], X[i:i+2], style, c=c, linewidth=lw, markersize=ms)
        
    return ax


def trajectories(X, ax=None, style='o', node_feature=None, lw=1, ms=5, arrowhead=1, axis=False, alpha=None):
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
            
    c = set_colors(node_feature)
    if alpha is not None:
        al=np.ones(len(X))*alpha
    elif len(c)>1 and not isinstance(c,str):
        al=np.abs(node_feature)/np.max(np.abs(node_feature))
    else:
        al=1
                
    if dim==2:
        if 'o' in style:
            ax.scatter(X[:, 0], X[:, 1], c=c, s=ms, alpha=al)
        if '-' in style:
            ax.plot(X[:, 0], X[:, 1], c=c, linewidth=lw, markersize=ms, alpha=al)
        if '>' in style:
            arrow_prop_dict = dict(color=c, alpha=al, lw=lw)
            for j in range(X.shape[0]):
                if j>0:
                    a = ax.arrow(X[j,0], X[j,1], X[j,0]-X[j-1,0], X[j,1]-X[j-2,1],
                                 **arrow_prop_dict)
                    ax.add_artist(a)
    elif dim==3:
        if 'o' in style:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c, s=ms, alpha=al)
        if '-' in style:
            ax.plot(X[:, 0], X[:, 1], X[:, 2], c=c, linewidth=lw, markersize=ms,alpha=al)
        if '>' in style:
            arrow_prop_dict = dict(mutation_scale=arrowhead, arrowstyle='-|>', color=c, alpha=al, lw=lw)
            for j in range(X.shape[0]):
                if j>0:
                    a = Arrow3D([X[j-1,0], X[j,0]], [X[j-1,1], X[j,1]], 
                                [X[j-1,2], X[j,2]], 
                                 **arrow_prop_dict)
                    ax.add_artist(a)
                
    if not axis:
        ax = set_axes(ax, data=None, off=True)
        
    return ax


def neighbourhoods(data,
                   clusters, 
                   n_samples=4,
                   radius=1,
                   norm=False, 
                   vector=False, 
                   plot_graph=False):
    
    nc = clusters['n_clusters']
    fig = plt.figure(figsize=(10, 20), constrained_layout=True)
    outer = gridspec.GridSpec(int(np.ceil(nc/3)), 3, wspace=0.2, hspace=0.2)
    
    data_list = data.to_data_list()
    graphs = [to_networkx(d, node_attrs=['pos'], edge_attrs=None, to_undirected=True,
            remove_self_loops=True) for d in data_list]
    
    node_values = [d.x for d in data_list]
    
    for i in range(nc):
        inner = gridspec.GridSpecFromSubplotSpec(int(np.ceil(n_samples/2)), 2,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.1)

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
            node_ids = nx.ego_graph(G,random_node,radius=radius).nodes
            node_ids = np.sort(node_ids) #sort nodes
            
            #if vector take magnitude
            if vector:
                vx = [nv[i,0] for i in node_ids]
                vy = [nv[i,1] for i in node_ids]
                nv = np.sqrt(nv[:,0]**2 + nv[:,1]**2)
                
            #convert node values to colors
            if not norm: #set colors based on global values
                c=set_colors(nv, cbar=False)
                c=[c[i] for i in node_ids] if isinstance(c,list) else c
            else: #first extract subgraph, then compute normalized colors
                nv=nv[node_ids]
                nv-=nv.mean()
                c=set_colors(nv, cbar=False)
                  
            ax = plt.Subplot(fig, inner[j])
            
            #extract subgraph with nodes sorted
            subgraph = nx.Graph()
            subgraph.add_nodes_from(sorted(G.subgraph(node_ids).nodes(data=True)))
            subgraph.add_edges_from(G.subgraph(node_ids).edges(data=True))
            
            ax.set_aspect('equal', 'box')
            if plot_graph:
                graph(subgraph,
                      node_values=None,
                      ax=ax,
                      node_size=30,
                      edge_width=0.5)
            
            x = np.array(list(nx.get_node_attributes(subgraph,name='pos').values()))
            if vector:
                ax.quiver(x[:,0],x[:,1],vx,vy, 
                          color=c, scale=10, scale_units='x', width=0.02)  
            else:
                ax.scatter(x[:,0],x[:,1],c=c)
            
            ax.set_frame_on(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)


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


def set_colors(color, cbar=True):
    
    if color is None:
        colors = 'C0'
    else:
        if isinstance(color, (list, tuple, np.ndarray)):
            cmap = plt.cm.coolwarm
            if (color>=0).all():
                norm = plt.cm.colors.Normalize(0, np.max(np.abs(color)))
            else:    
                norm = plt.cm.colors.Normalize(-np.max(np.abs(color)), np.max(np.abs(color)))
            if cbar:
                cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                plt.colorbar(cbar)
            colors = []
            for i, c in enumerate(color):
                colors.append(cmap(norm(np.array(c).flatten())))
        else:
            colors = color
            
    return colors


def graph(
    G,
    node_values='b',
    edge_width=1,
    edge_alpha=0.2,
    node_size=20,
    show_colorbar=False,
    layout=None,
    ax=None,
    node_attr="pos"
):
    """Plot scalar values on graph nodes embedded in 2D or 3D."""
        
    G = nx.convert_node_labels_to_integers(G)
    pos = list(nx.get_node_attributes(G, node_attr).values())
    
    if pos==[]:
        if layout=='spectral':
            pos = nx.spectral_layout(G)
        else:   
            pos = nx.spring_layout(G)
            
    dim = len(pos[0])
    assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
    if ax is None:
        _, ax = create_axis(dim)
    
    if len(pos[0])==2:
    
        if node_values is not None:
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                node_size=node_size,
                node_color=node_values,
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
    
    elif len(pos[0])==3:
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    
        if node_values is not None:
            ax.scatter(*node_xyz.T, s=node_size, ec="w")
        
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray", alpha=edge_alpha, linewidth=edge_width)      
            
    return ax
    

def embedding(emb, clusters, node_colors=None, titles=None):
        
    fig, ax = plt.subplots()
    
    colors = [f"C{i}" for i in np.arange(1, node_colors.max()+1)]
    cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(1, node_colors.max()+2), colors)
        
    scatter = ax.scatter(emb[:,0], emb[:,1], c=node_colors, alpha=0.3, cmap=cmap, norm=norm)
    vor = Voronoi(clusters['centroids']) 
    voronoi_plot_2d(vor, ax=ax, show_vertices=False) 
    handles,_ = scatter.legend_elements()
    
    #annotate clusters
    for k in range(clusters['n_clusters']):
        ax.annotate(k+1,clusters['centroids'][k,:])
    
    #add legend
    if titles is not None:
        ax.legend(handles,titles)
        
    ax.set_axis_off() 
        
        
def histograms(data, clusters, titles=None):
        
    l = clusters['labels']
    nc = clusters['n_clusters']
    slices = data._slice_dict['x']
    n_slices = len(slices)-1
    counts = []
    for i in range(n_slices):
        counts.append(l[slices[i]:slices[i+1]]+1)
        
    bins = [i+1 for i in range(nc)]
    
    fig = plt.figure(figsize=(10, 10),constrained_layout=True)
    grid = gridspec.GridSpec(int(np.ceil(n_slices//2)), 2, wspace=0.2, hspace=0.2)  
    
    for i in range(n_slices):
        ax = plt.Subplot(fig, grid[i])
        ax.hist(counts[i], bins=np.arange(nc+1)+0.5, rwidth=0.85, density=True)
        ax.set_xticks(bins)
        ax.set_xlim([0,nc+1])
        if titles is not None:
            ax.set_title(titles[i])
            ax.set_xlabel('Feature number')
            ax.set_ylabel('Probability density')
        fig.add_subplot(ax)


def _savefig(fig, folder, filename, ext):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        if not Path(folder).exists():
            os.mkdir(folder)
        fig.savefig((Path(folder) / filename).with_suffix(ext), bbox_inches="tight")
        
        

# def transition_diagram(centers, P, ax=None, radius=None, lw=1, ms=1, alpha=0.3, exclude_zeros=False):
    
#     dim = centers.shape[1]
#     assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
#     if ax is None:
#         _, ax = create_axis(dim)
        
#     colors = set_colors(P)
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
        

# def plot_curvatures(
#     times,
#     kappas,
#     ylog=True,
#     folder="figures",
#     filename="curvature",
#     ext=".svg",
#     ax=None
# ):
#     """Plot edge curvature."""
#     if ax is None:
#         fig, ax = create_axis(2)

#     for kappa in kappas.T:
#         ax.plot(times, kappa, c='k', lw=0.5, alpha=0.1)

#     if ylog:
#         ax.set_xscale("symlog")
        
#     ax.axhline(0, ls="--", c="k")
#     # ax.axis([np.log10(times[0]), np.log10(times[-1]), np.min(kappas), 1])
#     ax.set_xlabel(r"Time horizon, $log_{10}(T)$")
#     if ylog:
#         ax.set_ylabel(r"Curvature, $log_{10}\kappa_t(T)$")
#     else:
#         ax.set_ylabel(r"Curvature, $\kappa_t(T)$")
    
#     _savefig(fig, folder, filename, ext=ext)
    
#     return fig, ax
    
    
# def cuboid_data2(o, size=(1,1,1)):
    
#     X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
#          [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
#          [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
#          [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
#          [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
#          [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    
#     X = np.array(X).astype(float)
#     for i in range(3):
#         X[:,:,i] *= size[i]
#     X += np.array(o)
    
#     return X


# def plotCubeAt2(centers,sizes=None,colors=None, **kwargs):
    
#     if not isinstance(colors,(list,np.ndarray)):
#         colors=["C7"]*len(centers)
#     if not isinstance(sizes,(list,np.ndarray)):
#         sizes=[(1,1,1)]*len(centers)
        
#     for i in range(centers.shape[0]):
#         centers[i]-=sizes[i]/2
    
#     g = []
#     for p,s,c in zip(centers,sizes,colors):
#         g.append( cuboid_data2(p, size=s) )
        
#     return Poly3DCollection(np.concatenate(g),  
#                             facecolors=np.repeat(colors,6), **kwargs)


# def discretisation(centers, sizes, ax=None, alpha=0.2):
#     """
#     Plot the tesselation of the state space as a set of boxes.
#     """
        
#     dim = centers.shape[1]
#     assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
#     if ax is None:
#         _, ax = create_axis(dim)
    
#     pc = plotCubeAt2(centers,sizes,colors=None, edgecolor="k", linewidths=0.2, alpha=alpha)
#     ax.add_collection3d(pc)
    
#     ax = set_axes(ax, data=centers, off=True)
        
#     return ax