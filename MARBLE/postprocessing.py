"""Postprocessing module."""
import numpy as np

from MARBLE import geometry as g


def cluster(data, cluster_typ="kmeans", n_clusters=15, seed=0):
    clusters = g.cluster(data.emb, cluster_typ, n_clusters, seed)
    clusters = g.relabel_by_proximity(clusters)

    clusters["slices"] = data._slice_dict["x"]  # pylint: disable=protected-access

    if data.number_of_resamples > 1:
        clusters["slices"] = clusters["slices"][:: data.number_of_resamples]

    data.clusters = clusters

    return data


def distribution_distances(data, cluster_typ="kmeans", n_clusters=None, seed=0):
    """Return distance between datasets.

    Returns:
        data: PyG data object containing .out attribute, a nx2 matrix of embedded data
        clusters: sklearn cluster object
        dist (cxc matrix): pairwise distances where c is the number of clusters

    """

    emb = data.emb

    if n_clusters is not None:
        # k-means cluster
        data = cluster(data, cluster_typ, n_clusters, seed)

        # compute distances between clusters
        data.dist, data.gamma = g.compute_distribution_distances(
            clusters=data.clusters, slices=data.clusters["slices"]
        )

    else:
        data.emb = emb
        data.dist, _ = g.compute_distribution_distances(
            data=data, slices=data._slice_dict["x"]  # pylint: disable=protected-access
        )

    return data


def embed_in_2D(data, embed_typ="umap", manifold=None, seed=0):
    """Embed into 2D via for visualisation.

    Args:
        data: PyG input data
        embed_typl (string, optional): Embedding algorithm to use (tsne, umap, PCA)
        manifold (sklearn object, optional): Manifold object returned by some embedding algorithms
            (PCA, umap). Useful when trying to compare datasets.
        seed (int, optional): Random seed. The default is 0.

    Returns:
        PyG data object containing emb_2D attribute.
    """
    if isinstance(data, list):
        emb = np.vstack([d.emb for d in data])
    else:
        emb = data.emb

    if hasattr(data, "clusters"):
        clusters = data.clusters
        emb = np.vstack([emb, clusters["centroids"]])
        emb_2D, data.manifold = g.embed(emb, embed_typ=embed_typ, manifold=manifold, seed=seed)
        data.emb_2D, clusters["centroids"] = (
            emb_2D[: -clusters["n_clusters"]],
            emb_2D[-clusters["n_clusters"] :],
        )

    else:
        data.emb_2D, data.manifold = g.embed(emb, embed_typ=embed_typ, manifold=manifold, seed=seed)

    return data


# def compare_attractors(data, source_target):
#     """Compare attractors."""
#     assert all(
#         hasattr(data, attr) for attr in ["emb", "gamma", "clusters", "cdist"]
#     ), "It looks like postprocessing has not been run..."

#     s, t = source_target
#     slices = data._slice_dict["x"]  # pylint: disable=protected-access
#     n_slices = len(slices) - 1
#     s_s = range(slices[s], slices[s + 1])
#     s_t = range(slices[t], slices[t + 1])

#     assert s < n_slices - 2 and t < n_slices - 1, "Source and target must be < number of slices!"
#     assert s != t, "Source and target must be different!"

#     _, ax = plt.subplots(1, 3, figsize=(10, 5))

#     # plot embedding of all points in gray
#     plotting.embedding(data.emb, ax=ax[0], alpha=0.05)

#     # get gamma matrix for the given source-target pair
#     gammadist = data.gamma[s, t, ...]
#     np.fill_diagonal(gammadist, 0.0)

#     # color code source features
#     c = gammadist.sum(1)
#     cluster_ids = set(data.clusters["labels"][s_s])
#     labels = list(s_s)
#     for cid in cluster_ids:
#         idx = np.where(cid == data.clusters["labels"][s_s])[0]
#         for i in idx:
#             labels[i] = c[cid]

#     # plot source features in red
#     plotting.embedding(data.emb_2d[s_s], labels=labels, ax=ax[0], alpha=1.0)
#     prop_dict = {"style": ">", "lw": 2}
#     plotting.trajectories(data.pos[s_s], data.x[s_s], ax=ax[1], node_feature=labels, **prop_dict)
#     ax[1].set_title("Before")

#     # color code target features
#     c = gammadist.sum(0)
#     cluster_ids = set(data.clusters["labels"][s_t])
#     labels = list(s_t)
#     for cid in cluster_ids:
#         idx = np.where(cid == data.clusters["labels"][s_t])[0]
#         for i in idx:
#             labels[i] = -c[cid]  # negative for blue color

#     # plot target features in blue
#     plotting.embedding(data.emb_2d[s_t], labels=labels, ax=ax[0], alpha=1.0)
#     plotting.trajectories(data.pos[s_t], data.x[s_t], ax=ax[2], node_feature=labels, **prop_dict)
#     ax[2].set_title("After")
