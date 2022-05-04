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
    elif len(color)>1 and not isinstance(color,str):
        al=np.abs(node_feature)/np.max(np.abs(node_feature))
    else:
        al=1
                
    if dim==2:
        ax.scatter(X[:, 0], X[:, 1], c=color, s=ms, alpha=al)
        # ax.plot(X_l[:, 0], X_l[:, 1], style, c=c, linewidth=lw, markersize=ms, alpha=al)
        if style=='-':               
            for j in range(X.shape[0]):
                if j>0:
                    a = ax.arrow(X[j,0], X[j,1], X[j,0]-X[j-1,0], X[j,1]-X[j-2,1],
                                 lw=lw, arrowstyle="-|>", color=color, alpha=al)
                    ax.add_artist(a)
    elif dim==3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=ms, alpha=al)
        # ax.plot(X_l[:, 0], X_l[:, 1], X_l[:, 2], style, c=c, linewidth=lw, markersize=ms,alpha=al)
        if style=='-':
            for j in range(X.shape[0]):
                if j>0:
                    a = Arrow3D([X[j-1,0], X[j,0]], [X[j-1,1], X[j,1]], 
                                [X[j-1,2], X[j,2]], mutation_scale=ms, 
                                 lw=lw, arrowstyle="-|>", color=color, alpha=al)
                    ax.add_artist(a)
                
    if not axis:
        ax = set_axes(ax, data=None, off=True)
        
    return ax


def neighbourhoods(graphs, node_values, n_clusters, labels, n_samples=4, norm=True):
    fig = plt.figure(figsize=(10, 20),constrained_layout=True)
    outer = gridspec.GridSpec(int(np.ceil(n_clusters/3)), 3, wspace=0.2, hspace=0.2)
    
    for i in range(n_clusters):
        inner = gridspec.GridSpecFromSubplotSpec(int(np.ceil(n_samples/2)), 2,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        ax = plt.Subplot(fig, outer[i])
        ax.set_title("Type {}".format(i+1))
        ax.axis('off')
        fig.add_subplot(ax)
        
        n_nodes = [0] + [nx.number_of_nodes(g) for g in graphs]
        n_nodes = np.cumsum(n_nodes)

        for j, G in enumerate(graphs):
            
            label_i = labels[n_nodes[j]:n_nodes[j+1]]==i
            label_i = np.where(label_i)[0]
            if not list(label_i):
                continue
            else:
                random_node = np.random.choice(label_i)
            
            nv = node_values[j].numpy()
            ind_subgraph = [random_node] + list(G.neighbors(random_node))
            ind_subgraph = np.sort(ind_subgraph) #sort nodes
            
            #convert node values to colors
            if not norm: #set colors based on global values
                c=set_colors(nv, cbar=False)
                c=[c[i] for i in ind_subgraph]
            else: #first extract subgraph, then compute normalized colors
                nv=nv[ind_subgraph]
                nv-=nv.mean()
                c=set_colors(nv, cbar=False)
                  
            ax = plt.Subplot(fig, inner[j])
            
            #extract subgraph with nodes sorted
            subgraph = nx.Graph()
            subgraph.add_nodes_from(sorted(G.subgraph(ind_subgraph).nodes(data=True)))
            subgraph.add_edges_from(G.subgraph(ind_subgraph).edges(data=True))
            
            ax.set_aspect('equal', 'box')
            graph(subgraph,
                  node_colors=c,
                  show_colorbar=False,
                  ax=ax,
                  node_size=30,
                  edge_width=0.5)     
            ax.set_frame_on(False)
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
        colors = ['C0']
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
    edge_width=1,
    node_colors=None,
    node_size=20,
    show_colorbar=True,
    layout=None,
    ax=None,
    node_attr="x"
):
    """Plot the curvature on the graph."""
        
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
    
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            node_size=node_size,
            node_color=node_colors,
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_edges(
            G,
            pos=pos,
            width=edge_width,
            alpha=0.5,
            ax=ax
        )
    
    elif len(pos[0])==3:
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    
        # ax.scatter(*node_xyz.T, s=node_size, ec="w")
        
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")            
    

def embedding(emb, kmeans, node_colors=None, titles=None):
    from sklearn.manifold import TSNE
    
    # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=5, cmap='summer')
    
    fig, ax = plt.subplots()
    
    colors = [f"C{i}" for i in np.arange(1, node_colors.max()+1)]
    cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(1, node_colors.max()+2), colors)

    emb = emb.detach().numpy()
    n_emb = emb.shape[0]
    emb = np.vstack([emb, kmeans.cluster_centers_])
    
    if emb.shape[1]>2:
        emb = TSNE(init='random',learning_rate='auto').fit_transform(emb)
        
    x, y = emb[:n_emb,0], emb[:n_emb,1]
        
    scatter = ax.scatter(x, y, c=node_colors, alpha=0.3, cmap=cmap, norm=norm)
    vor = Voronoi(emb[n_emb:,:]) 
    voronoi_plot_2d(vor, ax=ax, show_vertices=False) 
    handles,_ = scatter.legend_elements()
    if titles is not None:
        ax.legend(handles,titles)
        
    ax.set_axis_off() 
        
        
def histograms(labels, slices, titles=None):
    fig = plt.figure(figsize=(10, 10),constrained_layout=True)
    
    n_clusters = np.max(labels)+1
    n_slices = len(slices['x'])-1
    counts = []
    for i in range(n_slices):
        counts.append(labels[slices['x'][i]:slices['x'][i+1]]+1)
        
    bins = [i+1 for i in range(n_clusters)]
    
    outer = gridspec.GridSpec(int(np.ceil(n_slices//2)), 2, wspace=0.2, hspace=0.2)
    
    for i in range(n_slices):

        ax = plt.Subplot(fig, outer[i])
        ax.hist(counts[i], bins=np.arange(n_clusters)-0.5, rwidth=0.85, density=True)
        ax.set_xticks(bins)
        ax.set_xlim([0,n_clusters+1])
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