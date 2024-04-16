"""Postprocessing module."""
import numpy as np

from MARBLE import geometry as g


def cluster(data, cluster_typ="kmeans", n_clusters=15, seed=0):
    """Cluster data."""
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

        
def rotate_systems(model, data): 
    """ Performs final learnt transformation to each vector field """  
       
    desired_layers = ['orthogonal']
    rotations_learnt = [param for i, (name, param) in enumerate(model.named_parameters()) if any(layer in name for layer in desired_layers)]
    
    data_list = []
    for i, d in enumerate(data.to_data_list()):
        p = d.pos
        x = d.x               
        rotation = rotations_learnt[i].cpu().detach().numpy() 
        d.pos_rotated = p @ rotation.T 
        d.x_rotated = x @ rotation.T 
        data_list.append(d)
        
    data_ = data.from_data_list(data_list)
    data.x_rotated = data_.x_rotated
    data.pos_rotated = data_.pos_rotated
        
    return data
