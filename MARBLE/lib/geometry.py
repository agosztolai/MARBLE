#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import scipy.sparse as sp

import torch_geometric.utils as PyGu
from torch_geometric.nn import knn_graph, radius_graph
from torch_scatter import scatter_add
from cknn import cknneighbors_graph

from torch.nn.functional import normalize

from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

from ptu_dijkstra import tangent_frames, connections
import ot

from . import utils

# =============================================================================
# Sampling
# =============================================================================
def sample_2d(N=100, interval=[[-1,-1],[1,1]], method='uniform', seed=0):
    """Sample N points in a 2D area."""
    if method=='uniform':
        x = np.linspace(interval[0][0], interval[1][0], int(np.sqrt(N)))
        y = np.linspace(interval[0][1], interval[1][1], int(np.sqrt(N)))
        x, y = np.meshgrid(x, y)
        x = np.vstack((x.flatten(), y.flatten())).T
        
    elif method=='random':
        np.random.seed(seed)
        x = np.random.uniform((interval[0][0], interval[0][1]), 
                              (interval[1][0], interval[1][1]), 
                              (N,2))
    
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
    
    if stop_crit==0.:
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
            if lambdas[i]/diam < stop_crit:
                perm = perm[:i]
                lambdas = lambdas[:i]
                break
        
    return perm, lambdas


# =============================================================================
# Clustering
# =============================================================================
def cluster(x, cluster_typ='meanshift', n_clusters=15, seed=0):
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
    
    clusters = dict()
    if cluster_typ=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(x)
        clusters['n_clusters'] = n_clusters
        clusters['labels'] = kmeans.labels_
        clusters['centroids'] = kmeans.cluster_centers_
    elif cluster_typ=='meanshift':
        meanshift = MeanShift(bandwidth=n_clusters).fit(x)
        clusters['n_clusters'] = len(set(meanshift.labels_))
        clusters['labels'] = meanshift.labels_
        clusters['centroids'] = meanshift.cluster_centers_
    else:
        NotImplementedError
        
    return clusters


def embed(x, embed_typ='umap', dim_emb=2, manifold=None):
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
    
    if x.shape[1]<=2:
        print('\n No {} embedding performed. Embedding seems to be \
              already in 2D.'.format(embed_typ))
        return x
    
    if embed_typ == 'tsne': 
        x = StandardScaler().fit_transform(x)
        if manifold is not None:
            raise Exception('t-SNE cannot fit on existing manifold')
            
        emb = TSNE(init='random',learning_rate='auto').fit_transform(x)
            
    elif embed_typ == 'umap':
        x = StandardScaler().fit_transform(x)
        if manifold is None:
            manifold = umap.UMAP().fit(x)
            
        emb = manifold.transform(x)
        
    elif embed_typ == 'MDS':
        if manifold is not None:
            raise Exception('MDS cannot fit on existing manifold')
            
        emb = MDS(n_components=dim_emb, n_init=20, dissimilarity='precomputed').fit_transform(x)
        
    elif embed_typ == 'PCA':
        if manifold is None:
            manifold = PCA(n_components=dim_emb).fit(x)
        
        emb = manifold.transform(x)
        
    else:
        NotImplementedError
        
    print('Performed {} embedding on embedded results.'.format(embed_typ))
    
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
    
    pd = pairwise_distances(clusters['centroids'], metric='euclidean')
    pd += np.max(pd)*np.eye(clusters['n_clusters'])
    
    mapping = dict()
    id_old = 0
    for i in range(clusters['n_clusters']):
        id_new = np.argmin(pd[id_old,:])
        while id_new in mapping.keys():
            pd[id_old,id_new] += np.max(pd)
            id_new = np.argmin(pd[id_old,:])
        mapping[id_new] = i
        id_old = id_new
        
    l = clusters['labels']
    clusters['labels'] = np.array([mapping[l[i]] for i,_ in enumerate(l)])
    clusters['centroids'] = clusters['centroids'][list(mapping.keys())]
    
    return clusters


def compute_histogram_distances(clusters):
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
    
    #compute discrete measures from clusters
    l, s = clusters['labels'], clusters['slices']
    l = [l[s[i]:s[i+1]]+1 for i in range(len(s)-1)]
    nc, nl = clusters['n_clusters'], len(l)
    bins_dataset = []
    for l_ in l: #loop over datasets
        bins = [(l_ == i+1).sum() for i in range(nc)] #loop over clusters
        bins = np.array(bins)
        bins_dataset.append(bins/bins.sum())
    
    #compute distance between measures
    dist = np.zeros([nl, nl])
    gamma = np.zeros([nl, nl, nc, nc])
    centroid_distances = pairwise_distances(clusters['centroids'])
    for i in range(nl):
        for j in range(i+1, nl):
            dist[i,j] = ot.emd2(bins_dataset[i], bins_dataset[j], centroid_distances)
            dist[j,i] = dist[i,j]
            gamma[i,j,...] = ot.emd(bins_dataset[i], bins_dataset[j], centroid_distances)
            gamma[j,i,...] = gamma[i,j,...]
                        
    return dist, gamma, centroid_distances

# def compute_distribution_distances(data, return_OT_matrix=True):
    
#     pdists = pairwise_distances(data.emb)
#     s = data._slice_dict['x']
#     nl = len(s)-1
    
#     dist = np.zeros([nl, nl])
#     gamma = [[[] for n in range(nl)] for n in range(nl)]
#     for i in range(nl):
#         for j in range(i+1, nl):
#             mu = np.ones(s[i+1]-s[i]); 
#             mu /= len(mu)
#             nu = np.ones(s[j+1]-s[j])
#             nu /= len(nu)
#             dxy = pdists[s[i]:s[i+1], s[j]:s[j+1]]
#             dist[i,j] = ot.emd2(mu, nu, dxy)
#             dist[j,i] = dist[i,j]
#             if return_OT_matrix:
#                 gamma[i][j].append(ot.emd(mu, nu, dxy))
#                 gamma[j][i].append(gamma[i][j])
                
#     if return_OT_matrix:
#         return dist, gamma
#     else:
#         return dist


# =============================================================================
# Manifold operations
# =============================================================================
def neighbour_vectors(pos, edge_index, normalise=False):
    """
    Local out-going edge vectors around each node.

    Parameters
    ----------
    pos : (nxdim) Matrix of node positions
    edge_index : (2x|E|) Matrix of edge indices
    normalise : If True then normalise neighbour vectors

    Returns
    -------
    nvec : (nxnxdim) Matrix of neighbourhood vectors.

    """
    
    n, dim = pos.shape
    
    nvec = pos.repeat(n,1,1)
    nvec = nvec - nvec.swapaxes(0,1) #nvec[i,j] = xj - xi
    
    mask = torch.zeros([n,n],dtype=bool)
    mask[edge_index[0], edge_index[1]] = 1
    mask = mask.unsqueeze(2).repeat(1,1,dim)
    nvec[~mask] = 0
    
    if normalise:
        nvec = normalize(nvec, dim=-1, p=2)
    
    return nvec


def map_to_local_gauges(x, gauges):
    """Transform signal into local coordinates"""
        
    return torch.einsum('aij,ai->aj', gauges, x)


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
    
    print('neighbours')
    nvec = neighbour_vectors(pos, edge_index, normalise=False) #(nxnxdim)
    print('project gauges')
    F = project_gauge_to_neighbours(nvec, gauges)
    
    K = []
    for _F in F:
        Fhat = normalize(_F, dim=1, p=1)
        Fhat -= torch.diag(torch.sum(Fhat, dim=1))
        K.append(Fhat.to_sparse())
            
    return K
    
    
def project_to_gauges(x, gauges, dim=2):
    coeffs = torch.einsum('bij,bi->bj', gauges, x)
    return torch.einsum('bj,bij->bi', coeffs[:,:dim], gauges[:,:,:dim])
    
    
def manifold_dimension(Sigma, frac_explained=0.9):
    """Estimate manifold dimension based on singular vectors"""
    
    if frac_explained==1.0:
        return Sigma.shape[1]
    
    Sigma **= 2
    Sigma /= Sigma.sum(1, keepdim=True)
    Sigma = Sigma.cumsum(dim=1)
    var_exp = Sigma.mean(0)-Sigma.std(0)
    dim_man = torch.where(var_exp >= frac_explained)[0][0] + 1
    
    print('\nFraction of variance explained: ', var_exp)
    
    return int(dim_man)


def project_gauge_to_neighbours(nvec, gauges):
    """
    Project the gauge vectors to local edge vectors.
    
    Parameters
    ----------
    nvec : (nxnxdim) Matrix of neighbourhood vectors.
    local_gauge : dimxnxdim torch tensor, if None, global gauge is generated

    Returns
    -------
    list of (nxn) torch tensors of projected components
    
    """
            
    #inner product over last dim of nvec and second dim of gauges
    #batch over first dims (a) and leave last broadcast over last dim of gauges (c)
    proj = torch.einsum('abi,aic->abc', nvec, gauges)
        
    return [proj[...,i] for i in range(proj.shape[-1])] #split into a list


def fit_graph(x, graph_type='cknn', par=1):
    """Fit graph to node positions"""
    
    if graph_type=='cknn':
        ckng = cknneighbors_graph(x, n_neighbors=par, delta=1.0)
        ckng += sp.eye(ckng.shape[0])
        edge_index = np.vstack(ckng.nonzero())
        edge_index = utils.np2torch(edge_index, dtype='double')
    elif graph_type=='knn':
        edge_index = knn_graph(x, k=par)
    elif graph_type=='radius':
        edge_index = radius_graph(x, r=par)
    else:
        NotImplementedError
    
    edge_index = PyGu.to_undirected(edge_index)
    pdist = torch.nn.PairwiseDistance(p=2)
    edge_weight = pdist(x[edge_index[0]], x[edge_index[1]])
    edge_weight = 1/edge_weight
    
    return edge_index, edge_weight


def compute_laplacian(data, normalization="rw"):
    
    edge_index, edge_attr = PyGu.get_laplacian(data.edge_index,
                           edge_weight = data.edge_weight,
                           normalization = normalization)
    
    return PyGu.to_dense_adj(edge_index, edge_attr=edge_attr).squeeze()


def compute_connection_laplacian(data, R, normalization='rw'):
    """
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
    
    n, d = data.x.shape
    
    #unnormalised (combinatorial) laplacian, to be normalised later
    L = compute_laplacian(data, normalization=None).to_sparse()
    
    #rearrange into block form (kron(L, ones(d,d)))
    edge_index = utils.expand_edge_index(L.indices(), dim=d)
    L = torch.sparse_coo_tensor(edge_index, L.values().repeat_interleave(d*d))
        
    #unnormalised connection laplacian 
    #Lc(i,j) = L(i,j)*R(i,j) if (i,j)=\in E else 0
    Lc = L*R
    
    #normalize
    edge_index, edge_weight = PyGu.remove_self_loops(data.edge_index, data.edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
    #degree matrix
    deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=n)
    
    if normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        deg_inv = deg_inv.repeat_interleave(d, dim=0)
        Lc = torch.diag(deg_inv).to_sparse()@Lc

    elif normalization == 'sym':
        NotImplementedError
        
    return Lc


def compute_gauges(data,
                           dim_man=None,
                           n_geodesic_nb=10,
                           n_workers=1
                           ):
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
    
    #make chunks for data processing
    sl = data._slice_dict['x']
        
    n = len(sl)-1
    X = [X[sl[i]:sl[i+1]] for i in range(n)]
    A = [A[sl[i]:sl[i+1],:][:,sl[i]:sl[i+1]] for i in range(n)]
    
    if dim_man is None:
        dim_man = X[0].shape[1]
        
    inputs = [X, A, dim_man, n_geodesic_nb]
    out = utils.parallel_proc(_compute_gauges, 
                              range(n), 
                              inputs,
                              processes=n_workers,
                              desc="Computing tangent spaces...")
        
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
    
    #make chunks for data processing
    sl = data._slice_dict['x']
    dim_man = gauges.shape[-1]
        
    n = len(sl)-1
    gauges = [gauges[sl[i]:sl[i+1]] for i in range(n)]
    A = [A[sl[i]:sl[i+1],:][:,sl[i]:sl[i+1]] for i in range(n)]
        
    inputs = [gauges, A, dim_man]
    out = utils.parallel_proc(_compute_connections, 
                              range(n), 
                              inputs,
                              processes=n_workers,
                              desc="Computing connections...")
    
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
def scalar_diffusion(x, t, method='matrix_exp', par=None):
    
    if len(x.shape)==1:
        x = x.unsqueeze(1)
    
    if method == 'matrix_exp':
        if par.is_sparse:
            par = par.to_dense()
        return torch.matrix_exp(-t*par.to_dense()).mm(x)
    
    if method == 'spectral':
        assert isinstance(par, (list, tuple)) and len(par)==2, 'For spectral method, par must be a tuple of \
            eigenvalues, eigenvectors!'
        evals, evecs = par

        # Transform to spectral
        x_spec = torch.mm(evecs.T, x)

        # Diffuse
        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * t.unsqueeze(0))
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex 
        return evecs.mm(x_diffuse_spec)

    else:
        NotImplementedError
    
    
def vector_diffusion(x, t, method='spectral', Lc=None, normalise=False):
    n, d = x.shape[0], x.shape[1]
    
    if method=='spectral':
        assert len(Lc)==2, 'Lc must be a tuple of eigenvalues, eigenvectors!'
        nd = Lc[0].shape[0]
    else:
        nd = Lc.shape[0]
    
    assert (n*d % nd)==0, \
        'Data dimension must be an integer multiple of the dimensions \
         of the connection Laplacian!'
        
    #vector diffusion with connection Laplacian
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
    
    
def compute_eigendecomposition(A, k=2, eps=1e-8):
    """
    Eigendecomposition of a square matrix A
    
    Parameters
    ----------
    A : square matrix A
    k : number of eigenvectors
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
            evals = torch.clamp(evals, min=0.)

            break
        except Exception as e:
            print(e)
            if(failcount > 3):
                raise ValueError("failed to compute eigendecomp")
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            A += torch.eye(A.shape[0]) * (eps * 10**(failcount-1))
    
    return evals, evecs