"""Plotting module."""
import os
from pathlib import Path
import torch
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import Voronoi
from scipy.spatial import voronoi_plot_2d
from torch_geometric.utils.convert import to_networkx

from .geometry import embed


def fields(
    data,
    titles=None,
    col=1,
    figsize=(8, 8),
    axlim=None,
    axes_visible=False,
    color=None,
    alpha=0.5,
    node_size=10,
    plot_gauges=False,
    width=0.005,
    edge_width=1.0,
    scale=5,
    view=None,
):
    """Plot scalar or vector fields

    Args:
        data: PyG Batch data object class created with utils.construct_dataset
        titles: list of titles
        col: int for number of columns to plot
        figsize: tuple of figure dimensions
    """
    if hasattr(data, "gauges"):
        gauges = data.gauges
    else:
        gauges = None

    if not isinstance(data, list):
        number_of_resamples = data.number_of_resamples
        data = data.to_data_list()  # split data batch

        if number_of_resamples > 1:
            print("\nDetected several samples. Taking only first one for visualisation!")
            data = data[::number_of_resamples]

    dim = data[0].pos.shape[1]
    vector = data[0].x.shape[1] > 1
    row = int(np.ceil(len(data) / col))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = gridspec.GridSpec(row, col, wspace=0.0, hspace=0.0, figure=fig)

    ax_list, lims = [], None
    for i, d in enumerate(data):
        signal = d.x.detach().numpy()
        _, ax = create_axis(dim, grid[i], fig=fig)

        if view is not None:
            ax.view_init(elev=view[0], azim=view[1])

        G = to_networkx(
            d, node_attrs=["pos"], edge_attrs=None, to_undirected=True, remove_self_loops=True
        )

        if color is None:
            c = np.linalg.norm(signal, axis=1) if vector else signal
            c, _ = set_colors(c.squeeze())
        else:
            c = color

        graph(
            G,
            labels=None if vector else c,
            ax=ax,
            node_size=node_size,
            edge_width=edge_width,
            edge_alpha=alpha,
            axes_visible=axes_visible,
        )

        if vector:
            pos = d.pos.numpy()
            plot_arrows(pos, signal, ax, c, scale=scale, width=width)

        if plot_gauges and (gauges is not None):
            for j in range(gauges.shape[2]):
                plot_arrows(pos, gauges[..., j], ax, "k", scale=scale)

        if titles is not None:
            ax.set_title(titles[i])

        fig.add_subplot(ax)

        if axlim is not None:
            if axlim == "same" and (lims is None):
                lims = get_limits(ax)
            elif len(axlim) == len(data):
                lims = axlim[i]
            else:
                raise NotImplementedError

        set_axes(ax, lims=lims, axes_visible=axes_visible)

        ax_list.append(ax)

    return ax_list


def histograms(data, titles=None, col=2, figsize=(10, 10)):
    """Plot histograms of cluster distribution across datasets.

    Args:
        data: PyG Batch data object class created with utils.construct_dataset
        clusters: sklearn cluster object
        titles: list of titles
        col: int for number of columns to plot
        figsize: tuple of figure dimensions
    """
    assert hasattr(
        data, "clusters"
    ), "No clusters found. First, run postprocessing.cluster(data)!"

    labels, s = data.clusters["labels"], data.clusters["slices"]
    n_slices = len(s) - 1
    labels = [labels[s[i] : s[i + 1]] + 1 for i in range(n_slices)]
    nc = data.clusters["n_clusters"]

    row = int(np.ceil(n_slices / col))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = gridspec.GridSpec(row, col, wspace=0.5, hspace=0.5, figure=fig)

    for i in range(n_slices):
        ax = plt.Subplot(fig, grid[i])

        ax.hist(labels[i], bins=np.arange(nc + 1) + 0.5, rwidth=0.85, density=True)
        ax.set_xticks(np.arange(nc) + 1)  # pylint: disable=not-callable
        ax.set_xlim([0, nc + 1])
        ax.set_xlabel("Feature number")
        ax.set_ylabel("Probability density")

        if titles is not None:
            ax.set_title(titles[i])

        fig.add_subplot(ax)


def embedding(
    data,
    labels=None,
    titles=None,
    mask=None,
    ax=None,
    alpha=0.3,
    s=5,
    axes_visible=False,
    cbar_visible=True,
    clusters_visible=False,
    cmap="coolwarm",
    plot_trajectories=False,
    style='o',
    lw=1,
    time_gradient=False
):
    """Plot embeddings.

    Args:
        data: PyG data object with attribute emb or nxdim matrix of embedded points with dim=2 or 3
        labels: list of increasing integer node labels
        clusters: sklearn cluster object
        titles: list of titles
    """
    if hasattr(data, "emb_2D"):
        emb = data.emb_2D
    elif isinstance(data, np.ndarray) or torch.is_tensor(data):
        emb = data

    dim = emb.shape[1]
    assert dim in [2, 3], f"Embedding dimension is {dim} which cannot be displayed."

    if ax is None:
        _, ax = create_axis(dim)

    if labels is not None:
        assert emb.shape[0] == len(labels)

    if labels is None:
        labels = np.ones(emb.shape[0])
        
    if mask is None:
        mask = np.ones(len(emb), dtype=bool)
        labels = labels[mask]

    types = sorted(set(labels))
    
    color, cbar = set_colors(types, cmap)
    
    if titles is not None:
        assert len(titles) == len(types)

    for i, typ in enumerate(types):
        title = titles[i] if titles is not None else str(typ)
        c_ = color[i]
        emb_ = emb[mask*(labels == typ)]
        
        if isinstance(data, np.ndarray) or torch.is_tensor(data):
            print('You need to pass a data object to plot trajectories!')
            plot_trajectories = False
        
        if plot_trajectories:
            l_ = data.l[mask*(labels == typ)]
            if len(l_) == 0:
                continue
            end = np.where(np.diff(l_)<0)[0]+1
            start = np.hstack([0, end])
            end = np.hstack([end, len(emb_)])
            cmap = LinearSegmentedColormap.from_list("Custom", [(0, 0, 0), c_], N=max(l_))
            
            for i, (s_,e_) in enumerate(zip(start, end)):
                t = range(s_,e_)
                cgrad = cmap(l_[t]/max(l_))
                if style=='-':
                    if time_gradient:
                        trajectories(emb_[t], style='-', ax=ax, ms=s, node_feature=cgrad, alpha=alpha, lw=lw)
                    else:
                        trajectories(emb_[t], style='-', ax=ax, ms=s, node_feature=[c_]*len(t), alpha=alpha, lw=lw)
                elif style=='o':
                    if dim == 2:
                        ax.scatter(emb_[t, 0], emb_[t, 1], c=cgrad, alpha=alpha, s=s, label=title)
                    elif dim == 3:
                        ax.scatter(emb_[t, 0], emb_[t, 1], emb_[t, 2], c=cgrad, alpha=alpha, s=s, label=title)  
        else:
            if dim == 2:
                ax.scatter(emb_[:, 0], emb_[:, 1], color=c_, alpha=alpha, s=s, label=title)
            elif dim == 3:
                ax.scatter(emb_[:, 0], emb_[:, 1], emb_[:, 2], color=c_, alpha=alpha, s=s, label=title)

    if dim == 2:
        if hasattr(data, "clusters") and clusters_visible:
            voronoi(data.clusters, ax)

    if titles is not None:
        ax.legend(loc="upper right")

    if not axes_visible:
        ax.set_axis_off()

    if cbar_visible and cbar is not None:
        plt.colorbar(cbar)

    return ax


def losses(model):
    """Model losses"""

    plt.plot(model.losses['train_loss'], label='Training loss')
    plt.plot(model.losses['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.legend()
    
    
def voronoi(clusters, ax):
    """Voronoi tesselation of clusters"""
    vor = Voronoi(clusters["centroids"])
    voronoi_plot_2d(vor, ax=ax, show_vertices=False)
    for k in range(clusters["n_clusters"]):
        ax.annotate(k + 1, clusters["centroids"][k, :])


def neighbourhoods(
    data,
    hops=1,
    cols=4,
    norm=False,
    color=None,
    plot_graph=False,
    figsize=(15, 20),
    fontsize=20,
    width=0.025,
    scale=1,
):
    """For each clustered neighbourhood type, draw one sample neighbourhood from each dataset.

    Args:
        data: postprocessed PyG Batch data object class created with utils.construct_dataset
        hops: size of neighbourhood in number of hops
        norm: if True, then normalise values to zero mean within clusters
        plot_graph: if True, then plot the underlying graph.
    """

    assert hasattr(
        data, "clusters"
    ), "No clusters found. First, run postprocessing.cluster(data)!"

    vector = data.x.shape[1] > 1
    clusters = data.clusters
    nc = clusters["n_clusters"]
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer = gridspec.GridSpec(int(np.ceil(nc / cols)), cols, wspace=0.2, hspace=0.2, figure=fig)

    number_of_resamples = data.number_of_resamples
    data = data.to_data_list()  # split data batch

    if number_of_resamples > 1:
        print(
            "\nDetected several samples of the same data. Taking only first one for visualisation!"
        )
        data = data[::number_of_resamples]

    graphs = []
    for d in data:
        graphs.append(
            to_networkx(
                d, node_attrs=["pos"], edge_attrs=None, to_undirected=True, remove_self_loops=True
            )
        )

    signals = [d.x for d in data]

    for i in range(nc):
        col = 2
        row = int(np.ceil(len(data) / col))
        inner = gridspec.GridSpecFromSubplotSpec(
            row, col, subplot_spec=outer[i], wspace=0.0, hspace=0.0
        )

        ax = plt.Subplot(fig, outer[i])
        ax.set_title(f"Type {i+1}", fontsize=fontsize)
        ax.axis("off")
        fig.add_subplot(ax)

        n_nodes = [0] + [nx.number_of_nodes(g) for g in graphs]
        n_nodes = np.cumsum(n_nodes)

        for j, G in enumerate(graphs):
            label_i = clusters["labels"][n_nodes[j] : n_nodes[j + 1]] == i
            label_i = np.where(label_i)[0]
            if not list(label_i):
                continue
            random_node = np.random.choice(label_i)

            signal = signals[j].numpy()
            node_ids = nx.ego_graph(G, random_node, radius=hops).nodes
            node_ids = np.sort(node_ids)  # sort nodes

            # convert node values to colors
            if color is not None:
                c = color
            else:
                c = signal
                if vector:
                    c = np.linalg.norm(signal, axis=1)

            if not norm:  # set colors based on global values
                c, _ = set_colors(c)
                c = [c[i] for i in node_ids] if isinstance(c, (list, np.ndarray)) else c
                signal = signal[node_ids]
            else:  # first extract subgraph, then compute normalized colors
                signal = signal[node_ids]
                signal -= signal.mean()
                c, _ = set_colors(signal.squeeze())

            ax = plt.Subplot(fig, inner[j])

            # extract subgraph with nodes sorted
            subgraph = nx.Graph()
            subgraph.add_nodes_from(sorted(G.subgraph(node_ids).nodes(data=True)))
            subgraph.add_edges_from(G.subgraph(node_ids).edges(data=True))

            ax.set_aspect("equal", "box")
            if plot_graph:
                graph(subgraph, labels=None, ax=ax, node_size=30, edge_width=0.5)

            pos = np.array(list(nx.get_node_attributes(subgraph, name="pos").values()))

            if pos.shape[1] > 2:
                pos, manifold = embed(pos, embed_typ="PCA")
                signal = embed(signal, embed_typ="PCA", manifold=manifold)[0]
            if vector:
                plot_arrows(pos, signal, ax, c, width=width, scale=scale)
            else:
                ax.scatter(pos[:, 0], pos[:, 1], c=c)

            ax.set_frame_on(False)
            set_axes(ax, axes_visible=False)
            fig.add_subplot(ax)


def graph(
    G,
    labels="b",
    edge_width=1,
    edge_alpha=1.0,
    node_size=20,
    layout=None,
    ax=None,
    axes_visible=True,
):
    """Plot scalar values on graph nodes embedded in 2D or 3D."""

    G = nx.convert_node_labels_to_integers(G)
    pos = list(nx.get_node_attributes(G, "pos").values())

    if not pos:
        if layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)

    dim = len(pos[0])
    assert dim in (2, 3), "Dimension must be 2 or 3."

    if ax is None:
        _, ax = create_axis(dim)

    if dim == 2:
        if labels is not None:
            nx.draw_networkx_nodes(
                G, pos=pos, node_size=node_size, node_color=labels, alpha=0.8, ax=ax
            )

        nx.draw_networkx_edges(G, pos=pos, width=edge_width, alpha=edge_alpha, ax=ax)

    elif dim == 3:
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

        if labels is not None:
            ax.scatter(*node_xyz.T, s=node_size, c=labels, ec="w")

        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray", alpha=edge_alpha, linewidth=edge_width)

    set_axes(ax, axes_visible=axes_visible)

    return ax


#def time_series(T, X, style="o", node_feature=None, figsize=(10, 5), lw=1, ms=5):
#    """Plot time series.

#    Args:
#        X (np array or list[np array]): Trajectories
#        style (string): Plotting style. The default is 'o'
#        color (bool): Color lines. The default is True
#        lw (int): Line width
#        ms (int): Marker size.

#    Returns:
#        matplotlib axes object
#    """
#    if not isinstance(X, list):
#        X = [X]

#    fig = plt.figure(figsize=figsize, constrained_layout=True)
#    grid = gridspec.GridSpec(len(X), 1, wspace=0.5, hspace=0, figure=fig)

#    for sp, X_ in enumerate(X):
#        if sp == 0:
#            ax = plt.Subplot(fig, grid[sp])
#        else:
#            ax = plt.Subplot(fig, grid[sp], sharex=ax)

#        ax.spines["top"].set_visible(False)
#        ax.spines["right"].set_visible(False)

#        if sp < len(X) - 1:
#            plt.setp(ax.get_xticklabels(), visible=False)  # pylint: disable=not-callable
#            ax.spines["bottom"].set_visible(False)
#            ax.xaxis.set_ticks_position("none")

#        colors = set_colors(node_feature)[0]

#        for i in range(len(X_) - 2):
#            if X_[i] is None:
#                continue

#            c = colors[i] if len(colors) > 1 and not isinstance(colors, str) else colors

#            ax.plot(T[i : i + 2], X_[i : i + 2], style, c=c, linewidth=lw, markersize=ms)

#            fig.add_subplot(ax)

#    return ax


def trajectories(
    X,
    V=None,
    ax=None,
    style="o",
    node_feature=None,
    lw=1,
    ms=5,
    scale=1,
    arrow_spacing=1,
    axes_visible=True,
    alpha=1.0,
):
    """Plot trajectory in phase space. If multiple trajectories are given, they are plotted with
    different colors.

    Args:
        X (np array): Positions
        V (np array): Velocities
        style (string): Plotting style. The default is 'o'
        node_feature: Color lines. The default is None
        lw (int): Line width
        ms (int): Marker size

    Returns:
        matplotlib axes object.
    """
    dim = X.shape[1]
    assert dim in (2, 3), "Dimension must be 2 or 3."

    if ax is None:
        _, ax = create_axis(dim)

    c = set_colors(node_feature)[0]
        
    if dim == 2:
        if "o" in style:
            ax.scatter(X[:, 0], X[:, 1], c=c, s=ms, alpha=alpha)
        if "-" in style:
            if isinstance(c, (np.ndarray, list, tuple)):
                for i in range(len(X) - 2):
                    ax.plot(
                        X[i : i + 2, 0],
                        X[i : i + 2, 1],
                        c=c[i],
                        linewidth=lw,
                        markersize=ms,
                        alpha=alpha,
                    )
            else:
                ax.plot(X[:, 0], X[:, 1], c=c, linewidth=lw, markersize=ms, alpha=alpha)
        if ">" in style:
            skip = (slice(None, None, arrow_spacing), slice(None))
            X, V = X[skip], V[skip]
            plot_arrows(X, V, ax, c, width=lw, scale=scale)

    elif dim == 3:
        if "o" in style:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c, s=ms, alpha=alpha)
        if "-" in style:
            if isinstance(c, (list, tuple)):
                for i in range(len(X) - 2):
                    ax.plot(
                        X[i : i + 2, 0],
                        X[i : i + 2, 1],
                        X[i : i + 2, 2],
                        c=c[i],
                        linewidth=lw,
                        markersize=ms,
                        alpha=alpha,
                        zorder=3,
                    )
            else:
                ax.plot(
                    X[:, 0],
                    X[:, 1],
                    X[:, 2],
                    c=c,
                    linewidth=lw,
                    markersize=ms,
                    alpha=alpha,
                    zorder=3,
                )
        if ">" in style:
            skip = (slice(None, None, arrow_spacing), slice(None))
            X, V = X[skip], V[skip]
            plot_arrows(X, V, ax, c, width=lw, scale=scale)

    set_axes(ax, axes_visible=axes_visible)

    return ax


def plot_arrows(pos, signal, ax, c="k", alpha=1.0, width=1.0, scale=1.0):
    """Plot arrows."""
    dim = pos.shape[1]
    if dim == 3:
        norm = signal.max() - signal.min()
        norm = norm if norm != 0 else 1
        scaling = (pos.max() - pos.min()) / norm / scale
        arrow_prop_dict = {
            "alpha": alpha,
            "mutation_scale": width,
            "arrowstyle": "-|>",
            "zorder": 3,
        }
        for j in range(len(pos)):
            a = Arrow3D(
                [pos[j, 0], pos[j, 0] + signal[j, 0] * scaling],
                [pos[j, 1], pos[j, 1] + signal[j, 1] * scaling],
                [pos[j, 2], pos[j, 2] + signal[j, 2] * scaling],
                **arrow_prop_dict,
                color=c[j] if len(c) > 1 else c,
            )
            ax.add_artist(a)

    if dim == 2:
        arrow_prop_dict = {"alpha": alpha, "zorder": 3, "scale_units": "inches"}
        ax.quiver(
            pos[:, 0],
            pos[:, 1],
            signal[:, 0],
            signal[:, 1],
            color=c if len(c) > 1 else c,
            scale=scale,
            width=width,
            **arrow_prop_dict,
        )


class Arrow3D(FancyArrowPatch):
    """Arrow 3D."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """draw."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self):
        """do 3d projection."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def create_axis(*args, fig=None):
    """Create axis."""
    dim = args[0]
    if len(args) > 1:
        args = [args[i] for i in range(1, len(args))]
    else:
        args = (1, 1, 1)

    if fig is None:
        fig = plt.figure()

    if dim == 2:
        ax = fig.add_subplot(*args)
    elif dim == 3:
        ax = fig.add_subplot(*args, projection="3d")

    return fig, ax


def get_limits(ax):
    """Get limits."""
    lims = [ax.get_xlim(), ax.get_ylim()]
    if ax.name == "3d":
        lims.append(ax.get_zlim())

    return lims


def set_axes(ax, lims=None, padding=0.1, axes_visible=True):
    """Set axes."""
    if lims is not None:
        xlim = lims[0]
        ylim = lims[1]
        pad = padding * (xlim[1] - xlim[0])

        ax.set_xlim([xlim[0] - pad, xlim[1] + pad])
        ax.set_ylim([ylim[0] - pad, ylim[1] + pad])
        if ax.name == "3d":
            zlim = lims[2]
            ax.set_zlim([zlim[0] - pad, zlim[1] + pad])

    if not axes_visible:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if ax.name == "3d":
            ax.set_zticklabels([])
        ax.axis("off")


def set_colors(color, cmap="coolwarm"):
    """Set colors."""
    if color is None:
        return "k", None

    if isinstance(color[0], (float, np.floating)):
        cmap = sns.color_palette(cmap, as_cmap=True)
        norm = plt.cm.colors.Normalize(0, np.max(np.abs(color)))
        colors = [cmap(norm(np.array(c).flatten())) for c in color]

    elif isinstance(color[0], (int, np.integer)):
        cmap = sns.color_palette()
        colors = [f"C{i}" for i in color]
        colors = [matplotlib.colors.to_rgba(c) for c in colors]
        cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(1, len(color) + 2), colors)
    elif isinstance(color[0], (np.ndarray, list, tuple)):
        return color, None
    else:
        raise Exception("color must be a list of integers or floats")

    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    return colors, cbar