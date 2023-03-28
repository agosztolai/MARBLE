"""Geometry module."""
import numpy as np
import ot
import scipy.sparse as sp
import torch
import torch_geometric.utils as PyGu
import umap
from cknn import cknneighbors_graph
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import knn_graph
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add

from ptu_dijkstra import connections  # isort:skip
from ptu_dijkstra import tangent_frames  # isort:skip

from . import utils  # isort:skip


# =============================================================================
# Sampling
# =============================================================================
def sample_2d(N=100, interval=None, method="uniform", seed=0):
    """Sample N points in a 2D area."""
    if interval is None:
        interval = [[-1, -1], [1, 1]]
    if method == "uniform":
        x = np.linspace(interval[0][0], interval[1][0], int(np.sqrt(N)))
        y = np.linspace(interval[0][1], interval[1][1], int(np.sqrt(N)))
        x, y = np.meshgrid(x, y)
        x = np.vstack((x.flatten(), y.flatten())).T

    elif method == "random":
        np.random.seed(seed)
        x = np.random.uniform(
            (interval[0][0], interval[0][1]), (interval[1][0], interval[1][1]), (N, 2)
        )

    return x


def furthest_point_sampling(x, N=None, stop_crit=0.1, start_idx=0):
    """
    A greedy O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    x : nxdim matrix of data
    N : Integer number of sampled points.
    stop_crit : when reaching this fraction of the total manifold diameter,
                we stop sampling
    start_idx : index of starting node

    Returns
    -------
    perm : node indices of the N sampled points
    lambdas : list of distances of furthest points

    """

    if stop_crit == 0.0:
        return torch.arange(len(x)), None

    D = utils.np2torch(pairwise_distances(x))
    n = D.shape[0] if N is None else N
    diam = D.max()

    start_idx = 5

    perm = torch.zeros(n, dtype=torch.int64)
    perm[0] = start_idx
    lambdas = torch.zeros(n)
    ds = D[start_idx, :]
    for i in range(1, n):
        idx = torch.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = torch.minimum(ds, D[idx, :])

        if N is None:
            if lambdas[i] / diam < stop_crit:
                perm = perm[:i]
                lambdas = lambdas[:i]
                break

    return perm, lambdas


# =============================================================================
# Clustering
# =============================================================================
def cluster(x, cluster_typ="meanshift", n_clusters=15, seed=0):
    """
    Cluster data

    Parameters
    ----------
    x : nxdim matrix of data
    cluster_typ : Clustering method.
    n_clusters : Number of clusters.
    seed

    Returns
    -------
    clusters : sklearn cluster object

    """

    clusters = {}
    if cluster_typ == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(x)
        clusters["n_clusters"] = n_clusters
        clusters["labels"] = kmeans.labels_
        clusters["centroids"] = kmeans.cluster_centers_
    elif cluster_typ == "meanshift":
        meanshift = MeanShift(bandwidth=n_clusters).fit(x)
        clusters["n_clusters"] = len(set(meanshift.labels_))
        clusters["labels"] = meanshift.labels_
        clusters["centroids"] = meanshift.cluster_centers_
    else:
        raise NotImplementedError

    return clusters


def embed(x, embed_typ="umap", dim_emb=2, manifold=None):
    """
    Embed data to 2D space.

    Parameters
    ----------
    x : nxdim matrix of data
    embed_typ : embedding method. The default is 'tsne'.

    Returns
    -------
    emb : nx2 matrix of embedded data

    """

    if x.shape[1] <= 2:
        print(
            f"\n No {embed_typ} embedding performed. Embedding seems to be \
              already in 2D."
        )
        return x

    if embed_typ == "tsne":
        x = StandardScaler().fit_transform(x)
        if manifold is not None:
            raise Exception("t-SNE cannot fit on existing manifold")

        emb = TSNE(init="random", learning_rate="auto", random_state=0).fit_transform(x)

    elif embed_typ == "umap":
        x = StandardScaler().fit_transform(x)
        if manifold is None:
            manifold = umap.UMAP(n_components=dim_emb, random_state=0).fit(x)

        emb = manifold.transform(x)

    elif embed_typ == "MDS":
        if manifold is not None:
            raise Exception("MDS cannot fit on existing manifold")

        emb = MDS(
            n_components=dim_emb, n_init=20, dissimilarity="precomputed", random_state=0
        ).fit_transform(x)

    elif embed_typ == "PCA":
        if manifold is None:
            manifold = PCA(n_components=dim_emb).fit(x)

        emb = manifold.transform(x)

    else:
        raise NotImplementedError

    print(f"Performed {embed_typ} embedding on embedded results.")

    return emb, manifold


def relabel_by_proximity(clusters):
    """
    Update clusters labels such that nearby clusters in the embedding get similar
    labels.

    Parameters
    ----------
    clusters : sklearn object containing 'centroids', 'n_clusters', 'labels'

    Returns
    -------
    clusters : sklearn object with updated labels

    """

    pd = pairwise_distances(clusters["centroids"], metric="euclidean")
    pd += np.max(pd) * np.eye(clusters["n_clusters"])

    mapping = {}
    id_old = 0
    for i in range(clusters["n_clusters"]):
        id_new = np.argmin(pd[id_old, :])
        while id_new in mapping:
            pd[id_old, id_new] += np.max(pd)
            id_new = np.argmin(pd[id_old, :])
        mapping[id_new] = i
        id_old = id_new

    labels = clusters["labels"]
    clusters["labels"] = np.array([mapping[label] for label in labels])
    clusters["centroids"] = clusters["centroids"][list(mapping.keys())]

    return clusters


def compute_distribution_distances(clusters=None, data=None, slices=None):
    """
    Compute the distance between clustered distributions across datasets.

    Parameters
    ----------
    clusters : clusters : sklearn object containing 'centroids', 'slices', 'labels'

    Returns
    -------
    dist : distance matrix
    gamma : optimal transport matrix
    centroid_distances : distances between cluster centroids

    """

    s = slices

    if clusters is not None:
        # compute discrete measures supported on cluster centroids
        labels = clusters["labels"]
        labels = [labels[s[i] : s[i + 1]] + 1 for i in range(len(s) - 1)]
        nc, nl = clusters["n_clusters"], len(labels)
        bins_dataset = []
        for l_ in labels:  # loop over datasets
            bins = [(l_ == i + 1).sum() for i in range(nc)]  # loop over clusters
            bins = np.array(bins)
            bins_dataset.append(bins / bins.sum())

        cdists = pairwise_distances(clusters["centroids"])
        gamma = np.zeros([nl, nl, nc, nc])

    elif data is not None:
        # compute empirical measures from datapoints
        nl = len(s) - 1

        bins_dataset = []
        for i in range(nl):
            mu = np.ones(s[i + 1] - s[i])
            mu /= len(mu)
            bins_dataset.append(mu)

        pdists = pairwise_distances(data.emb)
    else:
        raise Exception("No input provided.")

    # compute distance between measures
    dist = np.zeros([nl, nl])
    for i in range(nl):
        for j in range(i + 1, nl):
            mu, nu = bins_dataset[i], bins_dataset[j]

            if data is not None:
                cdists = pdists[s[i] : s[i + 1], s[j] : s[j + 1]]

            dist[i, j] = ot.emd2(mu, nu, cdists)
            dist[j, i] = dist[i, j]

            if clusters is not None:
                gamma[i, j, ...] = ot.emd(mu, nu, cdists)
                gamma[j, i, ...] = gamma[i, j, ...]
            else:
                gamma = None

    return dist, gamma


# =============================================================================
# Manifold operations
# =============================================================================
def neighbour_vectors(pos, edge_index):
    """
    Local out-going edge vectors around each node.

    Parameters
    ----------
    pos : (nxdim) Matrix of node positions
    edge_index : (2x|E|) Matrix of edge indices

    Returns
    -------
    nvec : (|E|xdim) Matrix of neighbourhood vectors.

    """

    ei, ej = edge_index[0], edge_index[1]
    nvec = pos[ej] - pos[ei]

    return nvec


def project_gauge_to_neighbours(nvec, gauges, edge_index):
    """
    Project the gauge vectors to local edge vectors.

    Parameters
    ----------
    nvec : (|E|xdim) Matrix of neighbourhood vectors.
    local_gauge : dimxnxdim torch tensor, if None, global gauge is generated

    Returns
    -------
    list of (nxn) torch tensors of projected components

    """

    n, _, d = gauges.shape
    ei = edge_index[0]
    proj = torch.einsum("bi,bic->bc", nvec, gauges[ei])

    proj = [sp.coo_matrix((proj[:, i], (edge_index)), [n, n]).tocsr() for i in range(d)]

    return proj


def gradient_op(pos, edge_index, gauges):
    """
    Directional derivative kernel from Beaini et al. 2021.

    Parameters
    ----------
    pos : (nxdim) Matrix of node positions
    edge_index : (2x|E|) Matrix of edge indices
    gauge : List of orthonormal unit vectors

    Returns
    -------
    K : list of (nxn) Anisotropic kernels.

    """

    nvec = neighbour_vectors(pos, edge_index)
    F = project_gauge_to_neighbours(nvec, gauges, edge_index)

    K = []
    for _F in F:
        norm = np.repeat(np.add.reduceat(np.abs(_F.data), _F.indptr[:-1]), np.diff(_F.indptr))
        _F.data /= norm
        _F -= sp.diags(np.array(_F.sum(1)).flatten())
        _F = _F.tocoo()
        K.append(torch.sparse_coo_tensor(np.vstack([_F.row, _F.col]), _F.data.data))

    return K


def normalize_sparse_matrix(sp_tensor):
    """Normalize sparse matrix."""
    row_sum = sp_tensor.sum(axis=1)
    row_sum[row_sum == 0] = 1  # to avoid divide by zero
    sp_tensor = sp_tensor.multiply(1.0 / row_sum)

    return sp_tensor


def map_to_local_gauges(x, gauges, length_correction=False):
    """Transform signal into local coordinates"""

    proj = torch.einsum("aij,ai->aj", gauges, x)

    if length_correction:
        norm_x = x.norm(p=2, dim=1, keepdim=True)
        norm_proj = proj.norm(p=2, dim=1, keepdim=True)
        proj = proj / norm_proj * norm_x

    return proj


def project_to_gauges(x, gauges, dim=2):
    """Project to gauges."""
    coeffs = torch.einsum("bij,bi->bj", gauges, x)
    return torch.einsum("bj,bij->bi", coeffs[:, :dim], gauges[:, :, :dim])


def manifold_dimension(Sigma, frac_explained=0.9):
    """Estimate manifold dimension based on singular vectors"""

    if frac_explained == 1.0:
        return Sigma.shape[1]

    Sigma **= 2
    Sigma /= Sigma.sum(1, keepdim=True)
    Sigma = Sigma.cumsum(dim=1)
    var_exp = Sigma.mean(0) - Sigma.std(0)
    dim_man = torch.where(var_exp >= frac_explained)[0][0] + 1

    print("\nFraction of variance explained: ", var_exp)

    return int(dim_man)


def fit_graph(x, graph_type="cknn", par=1, delta=1.0):
    """Fit graph to node positions"""

    if graph_type == "cknn":
        edge_index = cknneighbors_graph(x, n_neighbors=par, delta=delta).tocoo()
        edge_index = np.vstack([edge_index.row, edge_index.col])
        edge_index = utils.np2torch(edge_index, dtype="double")

    elif graph_type == "knn":
        edge_index = knn_graph(x, k=par)
        edge_index = PyGu.add_self_loops(edge_index)[0]

    elif graph_type == "radius":
        edge_index = radius_graph(x, r=par)
        edge_index = PyGu.add_self_loops(edge_index)[0]

    else:
        raise NotImplementedError

    assert is_connected(edge_index), "Graph is not connected! Try increasing k."

    edge_index = PyGu.to_undirected(edge_index)
    pdist = torch.nn.PairwiseDistance(p=2)
    edge_weight = pdist(x[edge_index[0]], x[edge_index[1]])
    edge_weight = 1 / edge_weight

    return edge_index, edge_weight


# def find_nn(X, ind_query=None, nn=1, r=None, theiler=10, n_jobs=-1):
#     """
#     Find nearest neighbors of a point on the manifold

#     Parameters
#     ----------
#     ind_query : 2d np array, list[2d np array]
#         Index of points whose neighbors are needed.
#     x : nxd array (dimensions are columns!)
#         Coordinates of n points on a manifold in d-dimensional space.
#     nn : int, optional
#         Number of nearest neighbors. The default is 1.
#     theiler : int
#         Theiler exclusion. Do not include the points immediately before or
#         after in time the query point as neighbours.
#     n_jobs : int, optional
#         Number of processors to use. The default is all.

#     Returns
#     -------
#     dist : list[list]
#         Distance of nearest neighbors.
#     ind : list[list]
#         Index of nearest neighbors.

#     """

#     if ind_query is None:
#         ind_query = np.arange(len(X))
#     elif isinstance(ind_query, list):
#         ind_query = np.vstack(ind_query)

#     #Fit neighbor estimator object
#     kdt = KDTree(X, leaf_size=30, metric='euclidean')

#     inputs = [kdt, X, r, nn, theiler]
#     ind = utils.parallel_proc(nb_query,
#                         ind_query,
#                         inputs,
#                         desc="Computing neighbours...")

#     return ind


# def nb_query(inputs, i):

#     kdt, X, r, nn, theiler  = inputs

#     x_query = X[i]
#     if r is not None:
#         ind, dist = kdt.query_radius(x_query, r=r, return_distance=True, sort_results=True)
#         ind = ind[0]
#         dist = dist[0]
#     else:
#         # apparently, the outputs are reversed here compared to query_radius()
#         _, ind = kdt.query(x_query, k=nn+2*theiler+1)

#     #Theiler exclusion (points immediately before or after are not useful neighbours)
#     ind = ind[np.abs(ind-i)>theiler][:nn]

#     # edges = np.vstack()

#     return ind


def is_connected(edge_index):
    """Check if it is connected."""
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]))
    deg = torch.sparse.sum(adj, 0).values()

    return (deg > 1).all()


def compute_laplacian(data, normalization="rw"):
    """Compute Laplacian."""
    edge_index, edge_attr = PyGu.get_laplacian(
        data.edge_index, edge_weight=data.edge_weight, normalization=normalization
    )

    return PyGu.to_dense_adj(edge_index, edge_attr=edge_attr).squeeze()


def compute_connection_laplacian(data, R, normalization="rw"):
    r"""
    Connection Laplacian

    Parameters
    ----------
    data : Pytorch geometric data object.
    R : (nxnxdxd) Connection matrices between all pairs of nodes.
        Default is None, in case of a global coordinate system.
    normalization: None, 'sym', 'rw'
                 1. None: No normalization
                 :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

                 2. "sym"`: Symmetric normalization
                 :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
                 \mathbf{D}^{-1/2}`

                 3. "rw"`: Random-walk normalization
                 :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

    Returns
    -------
    (ndxnd) Normalised connection Laplacian matrix.

    """

    n = data.x.shape[0]
    d = R.size()[0] // n

    # unnormalised (combinatorial) laplacian, to be normalised later
    L = compute_laplacian(data, normalization=None).to_sparse()

    # rearrange into block form (kron(L, ones(d,d)))
    edge_index = utils.expand_edge_index(L.indices(), dim=d)
    L = torch.sparse_coo_tensor(edge_index, L.values().repeat_interleave(d * d))

    # unnormalised connection laplacian
    # Lc(i,j) = L(i,j)*R(i,j) if (i,j)=\in E else 0
    Lc = L * R

    # normalize
    edge_index, edge_weight = PyGu.remove_self_loops(data.edge_index, data.edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    # degree matrix
    deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=n)

    if normalization == "rw":
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        deg_inv = deg_inv.repeat_interleave(d, dim=0)
        Lc = torch.diag(deg_inv).to_sparse() @ Lc

    elif normalization == "sym":
        raise NotImplementedError

    return Lc


def compute_gauges(data, dim_man=None, n_geodesic_nb=10, n_workers=1):
    """
    Orthonormal gauges for the tangent space at each node, and connection
    matrices between each pair of adjacent nodes.

    R is a block matrix, where the row index is the gauge we want to align to,
    i.e., gauges(i) = R[i,j]@gauges(j).

    R[i,j] is optimal rotation that minimises ||X - RY||_F computed by SVD:
    X, Y = gauges[i].T, gauges[j].T
    U, _, Vt = scipy.linalg.svd(X.T@Y)
    R[i,j] = U@Vt

    Parameters
    ----------
    data : Pytorch geometric data object.
    n_geodesic_nb : number of geodesic neighbours. The default is 10.

    Returns
    -------
    gauges : (nxdimxdim) Matrix containing dim unit vectors for each node.
    Sigma : Singular valued
    R : (n*dimxn*dim) Connection matrices.

    """
    X = data.pos.numpy().astype(np.float64)
    A = PyGu.to_scipy_sparse_matrix(data.edge_index).tocsr()

    # make chunks for data processing
    sl = data._slice_dict["x"]  # pylint: disable=protected-access

    n = len(sl) - 1
    X = [X[sl[i] : sl[i + 1]] for i in range(n)]
    A = [A[sl[i] : sl[i + 1], :][:, sl[i] : sl[i + 1]] for i in range(n)]

    if dim_man is None:
        dim_man = X[0].shape[1]

    inputs = [X, A, dim_man, n_geodesic_nb]
    out = utils.parallel_proc(
        _compute_gauges, range(n), inputs, processes=n_workers, desc="Computing tangent spaces..."
    )

    gauges, Sigma = zip(*out)
    gauges, Sigma = np.vstack(gauges), np.vstack(Sigma)

    return utils.np2torch(gauges), utils.np2torch(Sigma)


def _compute_gauges(inputs, i):
    X_chunks, A_chunks, dim_man, n_geodesic_nb = inputs

    gauges, Sigma = tangent_frames(X_chunks[i], A_chunks[i], dim_man, n_geodesic_nb)

    return gauges, Sigma


def compute_connections(data, gauges, n_workers=1):
    """
    Find smallest rotations R between gauges pairs. It is assumed that the first
    row of edge_index is what we want to align to, i.e.,
    gauges(i) = gauges(j)@R[i,j].T

    Parameters
    ----------
    data : Pytorch geometric data object.
    gauges : (n,d,d) matrix of orthogonal unit vectors for each node

    Returns
    -------
    R : (n*dim,n*dim) matrix of rotation matrices

    """

    gauges = np.array(gauges, dtype=np.float64)
    A = PyGu.to_scipy_sparse_matrix(data.edge_index).tocsr()

    # make chunks for data processing
    sl = data._slice_dict["x"]  # pylint: disable=protected-access
    dim_man = gauges.shape[-1]

    n = len(sl) - 1
    gauges = [gauges[sl[i] : sl[i + 1]] for i in range(n)]
    A = [A[sl[i] : sl[i + 1], :][:, sl[i] : sl[i + 1]] for i in range(n)]

    inputs = [gauges, A, dim_man]
    out = utils.parallel_proc(
        _compute_connections, range(n), inputs, processes=n_workers, desc="Computing connections..."
    )

    return utils.to_block_diag(out)


def _compute_connections(inputs, i):
    gauges_chunks, A_chunks, dim_man = inputs

    R = connections(gauges_chunks[i], A_chunks[i], dim_man)

    edge_index = np.vstack([A_chunks[i].tocoo().row, A_chunks[i].tocoo().col])
    edge_index = torch.tensor(edge_index)
    edge_index = utils.expand_edge_index(edge_index, dim=R.shape[-1])
    R = torch.sparse_coo_tensor(edge_index, R.flatten(), dtype=torch.float32).coalesce()

    return R


# =============================================================================
# Diffusion
# =============================================================================
def scalar_diffusion(x, t, method="matrix_exp", par=None):
    """Scalar diffusion."""
    if len(x.shape) == 1:
        x = x.unsqueeze(1)

    if method == "matrix_exp":
        if par.is_sparse:
            par = par.to_dense()
        return torch.matrix_exp(-t * par.to_dense()).mm(x)

    if method == "spectral":
        assert (
            isinstance(par, (list, tuple)) and len(par) == 2
        ), "For spectral method, par must be a tuple of \
            eigenvalues, eigenvectors!"
        evals, evecs = par

        # Transform to spectral
        x_spec = torch.mm(evecs.T, x)

        # Diffuse
        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * t.unsqueeze(0))
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex
        return evecs.mm(x_diffuse_spec)

    raise NotImplementedError


def vector_diffusion(x, t, method="spectral", Lc=None):
    """Vector diffusion."""
    n, d = x.shape[0], x.shape[1]

    if method == "spectral":
        assert len(Lc) == 2, "Lc must be a tuple of eigenvalues, eigenvectors!"
        nd = Lc[0].shape[0]
    else:
        nd = Lc.shape[0]

    assert (
        n * d % nd
    ) == 0, "Data dimension must be an integer multiple of the dimensions \
         of the connection Laplacian!"

    # vector diffusion with connection Laplacian
    out = x.view(nd, -1)
    out = scalar_diffusion(out, t, method, Lc)
    out = out.view(x.shape)

    # if normalise:
    #     assert par['L'] is not None, 'Need Laplacian for normalised diffusion!'
    #     x_abs = x.norm(dim=-1, p=2, keepdim=True)
    #     out_abs = scalar_diffusion(x_abs, t, method, par['L'])
    #     ind = scalar_diffusion(torch.ones(x.shape[0],1), t, method, par['L'])
    #     out = out*out_abs/(ind*out.norm(dim=-1, p=2, keepdim=True))

    return out


def compute_eigendecomposition(A, eps=1e-8):
    """
    Eigendecomposition of a square matrix A

    Parameters
    ----------
    A : square matrix A
    eps : small error term

    Returns
    -------
    evals : (k) eigenvalues of the Laplacian
    evecs : (V,k) matrix of eigenvectors of the Laplacian

    """

    if A is None:
        return None

    A = A.to_dense()

    # Compute the eigenbasis
    failcount = 0
    while True:
        try:
            evals, evecs = torch.linalg.eigh(A)
            evals = torch.clamp(evals, min=0.0)

            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)
            if failcount > 3:
                raise ValueError("failed to compute eigendecomp") from e
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            A += torch.eye(A.shape[0]) * (eps * 10 ** (failcount - 1))

    return evals, evecs
