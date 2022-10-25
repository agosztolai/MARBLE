#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import scipy.sparse as sp
from scipy.linalg import orthogonal_procrustes

import torch_geometric.utils as PyGu
from torch_geometric.nn import knn_graph, radius_graph
from torch_scatter import scatter_add
from cknn import cknneighbors_graph

from torch.nn.functional import normalize

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
import umap

from ptu_dijkstra import ptu_dijkstra
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
        x = np.stack([x,y],axis=1)
        
    elif method=='random':
        np.random.seed(seed)
        x = np.random.uniform((interval[0][0], interval[0][1]), 
                              (interval[1][0], interval[1][1]), 
                              (N,2))
    
    return x


def furthest_point_sampling(x, N=None, stop_crit=0.1):
    """
    A greedy O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    x : nxdim matrix of data
    N : Integer number of sampled points.
    stop_crit : when reaching thisfraction of the total manifold diameter,
                we stop sampling

    Returns
    -------
    perm : node indices of the N sampled points
    lambdas : list of furthest points

    """
    
    D = pairwise_distances(x)
    n = D.shape[0] if N is None else N
    diam = D.max()
    
    perm = np.zeros(n, dtype=np.int64)
    lambdas = np.zeros(n)
    ds = D[0, :]
    for i in range(1, n):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
        
        if N is None:
            if lambdas[i]/diam < stop_crit:
                perm = perm[:i]
                lambdas = lambdas[:i]
                break
        
    return perm, lambdas


# =============================================================================
# Clustering
# =============================================================================
def cluster_embedding(data,
                      cluster_typ='kmeans', 
                      embed_typ='tsne', 
                      n_clusters=15, 
                      seed=0):
    """
    Cluster embedding and return distance between clusters
    
    Returns
    -------
    data : PyG data object containing .emb attribute, a nx2 matrix of embedded data
    clusters : sklearn cluster object
    dist : cxc matrix of pairwise distances where c is the number of clusters
    
    """

    emb = data.emb
    
    clusters = cluster(emb, cluster_typ, n_clusters, seed)
    clusters = relabel_by_proximity(clusters)
    clusters['slices'] = data._slice_dict['x']
    
    emb = np.vstack([emb, clusters['centroids']])
    emb = embed(emb, embed_typ)  
    emb, clusters['centroids'] = emb[:-n_clusters], emb[-n_clusters:]
    
    dist = compute_histogram_distances(clusters)
        
    return emb, clusters, dist


def cluster(x, cluster_typ='kmeans', n_clusters=15, seed=0):
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
    else:
        NotImplementedError
        
    return clusters


def embed(x, embed_typ='tnse'):
    """
    Embed data to Euclidean space

    Parameters
    ----------
    x : nxdim matrix of data
    embed_typ : embedding method. The default is 'tsne'.

    Returns
    -------
    emb : nx2 matrix of embedded data

    """
    
    if embed_typ == 'tsne': 
        if x.shape[1]>2:
            print('Performed t-SNE embedding on embedded results.')
            x = StandardScaler().fit_transform(x)
            emb = TSNE(init='random',learning_rate='auto').fit_transform(x)
            
    elif embed_typ == 'umap':
        print('Performed UMAP embedding on embedded results.')
        x = StandardScaler().fit_transform(x)
        emb = umap.UMAP().fit_transform(x)
        
    elif embed_typ == 'MDS':
        emb = MDS(n_components=2, dissimilarity='precomputed').fit_transform(x)
    else:
        NotImplementedError
    
    return emb


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


def compute_histogram_distances(clusters, dist_typ='Wasserstein'):
    
    l, s = clusters['labels'], clusters['slices']
    l = [l[s[i]:s[i+1]]+1 for i in range(len(s)-1)]
    nc, nl = clusters['n_clusters'], len(l)
    bins = []
    for l_ in l:
        bins_i = []
        for i in range(nc):
            bins_i.append((l_ == i).sum())
        bins_i = np.array(bins_i)
        bins.append(bins_i/bins_i.sum())
    
    dist = np.zeros([nl, nl])
    if dist_typ == 'Wasserstein':
        centroid_distances = pairwise_distances(clusters['centroids'])
        for i in range(nl):
            for j in range(i+1,nl):
                dist[i,j] = ot.emd2(bins[i], bins[j], centroid_distances)
                
        dist += dist.T
        
    elif dist_typ == 'KL_divergence':
        NotImplementedError
    else:
        NotImplementedError
    
    return dist


def compute_distribution_distances(data, dist_typ='Wasserstein'):
    
    pdists = pairwise_distances(data.emb)
    s = data._slice_dict['x']
    n = len(s)-1
    
    dist = np.zeros([n, n])
    for i in range(n):
        for j in range(i+1, n):
            mu = np.ones(s[i+1]-s[i])
            nu = np.ones(s[j+1]-s[j])
            dist[i,j] = ot.emd2(mu/len(mu), 
                                nu/len(nu), 
                                pdists[s[i]:s[i+1], s[j]:s[j+1]])
            
    dist += dist.T
    
    data._slice_dict['x']
    
    return dist


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


def optimal_rotation(X, Y):
    """Optimal rotation between orthogonal coordinate frames X and Y"""
    
    XtY = X.T@Y
    n = XtY[0].shape[0]
    U, S, Vt = np.linalg.svd(XtY)
    UVt = U @ Vt
    if abs(1.0 - np.linalg.det(UVt)) < 1e-10:
        return UVt
    # UVt is in O(n) but not SO(n), which is easily corrected.
    J = np.append(np.ones(n - 1), -1)
    return (U * J) @ Vt


# def gradient_op(pos, edge_index, gauges):
#     """Gradient operator

#     Parameters
#     ----------
#     nvec : (nxnxdim) Matrix of neighbourhood vectors.

#     Returns
#     -------
#     G : (nxn) Gradient operator matrix.

#     """

#     nvec = -neighbour_vectors(pos, edge_index, normalise=False) #(nxnxdim)
    
#     G = torch.zeros_like(nvec)
#     for i, g_ in enumerate(nvec):
#         nb_ind = torch.where(g_[:,0]!=0)[0]
#         A = g_[nb_ind]
#         B = torch.column_stack([-1.*torch.ones((len(nb_ind),1)),
#                                 torch.eye(len(nb_ind))])
#         grad = torch.linalg.lstsq(A, B).solution
#         G[i,i,:] = grad[:,[0]].T
#         G[i,nb_ind,:] = grad[:,1:].T
            
#     return [G[...,i] for i in range(G.shape[-1])]


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
    
    nvec = neighbour_vectors(pos, edge_index, normalise=False) #(nxnxdim)
    F = project_gauge_to_neighbours(nvec, gauges)
    
    K = []
    for _F in F:
        Fhat = normalize(_F, dim=-1, p=1)
        K.append(Fhat - torch.diag(torch.sum(Fhat, dim=1)))
            
    return K


def compute_gauges(data, local=True, n_nb=10):
    """
    Compute gauges

    Parameters
    ----------
    data : pytorch geometric data object containing .pos and .edge_index
    local : bool, The default is False.
    n_nb : int, Number of neighbours to use to compute gauges. Should be
    more than the number of neighbours in the knn graph. The default is 10.

    Returns
    -------
    gauges : (n,dim,dim) matric of orthogonal unit vectors for each node
    R : (n,n,dim,dim) connection matrices. If local=True.

    """
    
    n = data.pos.shape[0]
    dim = data.pos.shape[-1]
    
    if local:
        gauges, Sigma = compute_tangent_bundle(data, n_geodesic_nb=n_nb)
        return gauges, Sigma
    else:      
        gauges = torch.eye(dim)
        gauges = gauges.repeat(n,1,1)      
        return gauges, None
    
    
def manifold_dimension(Sigma, frac_explained=0.9):
    """Estimate manifold dimension based on singular vectors"""
    
    Sigma **= 2
    Sigma /= Sigma.sum(1, keepdim=True)
    Sigma = Sigma.cumsum(dim=1)
    dim_man = (Sigma<frac_explained).sum(0)
    dim_man = torch.where(dim_man<Sigma.shape[0]*(1-frac_explained))[0][0] + 1
    
    return dim_man


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
            
    gauges = gauges.swapaxes(0,1) #(nxdimxdim) -> (dimxnxdim)
        
    return [(nvec*g).sum(-1) for g in gauges] #dot product in last dimension


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
    L : (nxn) Laplacian matrix.
    R : (nxnxdimxdim) Connection matrices between all pairs of nodes.
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
    (n*dimxn*dim) Connection Laplacian matrox.

    """
    
    L = compute_laplacian(data, normalization=None)
    
    n = L.shape[0]
    dim = data.pos.shape[-1]
    
    #rearrange into block form
    L = torch.kron(L, torch.ones([dim,dim]))
    
    R = R.swapaxes(1,2).reshape(n*dim, n*dim)
    R += torch.eye(n).kron(torch.ones(dim,dim))
        
    #multiply off-diagonal terms
    Lc = L*R
       
    #normalize
    edge_index, edge_weight = PyGu.remove_self_loops(data.edge_index, data.edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    row, _ = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=data.num_nodes)
    
    if normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        deg_inv = deg_inv.repeat_interleave(dim, dim=0)
        Lc *= deg_inv

    elif normalization == 'sym':
        NotImplementedError
        
    return Lc


def compute_tangent_bundle(data, n_geodesic_nb=10, return_predecessors=True):
    """
    Orthonormal gauges for the tangent space at each node, and connection 
    matrices between each pair of adjacent nodes.

    Parameters
    ----------
    data : Pytorch geometric data object.
    n_geodesic_nb : number of geodesic neighbours. The default is 10.
    return_predecessors : bool. The default is True.

    Returns
    -------
    tangents : (nxdimxdim) Matrix containing dim unit vectors for each node.
    R : (nxnxdimxdim) Connection matrices between all pairs of nodes.

    """
    X = data.pos.numpy().astype(np.float64)
    A = PyGu.to_scipy_sparse_matrix(data.edge_index).tocsr()

    _, _, tangents, Sigma, _ = ptu_dijkstra(X, A, X.shape[1], n_geodesic_nb, return_predecessors)
    
    return utils.np2torch(tangents), utils.np2torch(Sigma)
    
    
def compute_connections(gauges, edge_index, dim_man=None):
    """
    Find smallest rotations R between gauges pairs. It is assumed that the first 
    row of edge_index is what we want to align to, i.e., 
    gauges(edge_index[1]) = R*gauges(edge_index[1]).

    Parameters
    ----------
    gauges : (n,dim,dim) matrix of orthogonal unit vectors for each node
    edge_index : (2x|E|) Matrix of edge indices
    dim_man : integer, manifold dimension

    Returns
    -------
    R : (n,n,dim,dim) matrix of rotation matrices

    """
    n, dim, k = gauges.shape
    
    R = np.eye(dim)[None,None,:,:]
    R = np.tile(R, (n,n,1,1))
    
    if dim_man is not None:
        if dim_man == dim: #the manifold is the whole space
            return R
        elif dim_man <= dim:
            gauges = gauges[:,:,dim_man:]
        else:
            raise Exception('Manifold dim must be <= embedding dim!') 
                
    for (i,j) in edge_index.T:
        R[i,j,...] = procrustes(gauges[i].T, gauges[j].T)

    return R


def procrustes(X, Y, reflection_allowed=False):
    """Solve for rotation that minimises ||X - RY||_F """

    # optimum rotation matrix of Y
    R = orthogonal_procrustes(X, Y)[0]

    # does the current solution use a reflection?
    reflection = np.linalg.det(R) < 0
    
    if reflection_allowed and reflection:
        R = np.zeros_like(R)
       
    return R


# =============================================================================
# Diffusion
# =============================================================================
def scalar_diffusion(x, t, method='matrix_exp', par=None):
    
    if len(x.shape)==1:
        x = x.unsqueeze(1)
    
    if method == 'matrix_exp':
        return torch.matrix_exp(-t*par['L']).mm(x)
    
    if method == 'spectral':
        
        evals, evecs = par['evals'], par['evecs'] 

        # Transform to spectral
        x_spec = torch.mm(evecs.T, x)

        # Diffuse
        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * t.unsqueeze(0))
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex 
        return evecs.mm(x_diffuse_spec)

    else:
        NotImplementedError
    
    
def vector_diffusion(x, t, method='spectral', par=None, normalise=False):
        
    #vector diffusion with connection Laplacian
    out = x.view(par['Lc'].shape[0], -1)
    out = scalar_diffusion(out, t, method, {'L': par['Lc']})
    out = out.view(x.shape)
    
    if normalise:
        assert par['L'] is not None, 'Need Laplacian for normalised diffusion!'
        x_abs = x.norm(dim=-1, p=2, keepdim=True)
        out_abs = scalar_diffusion(x_abs, t, method, {'L': par['L']})
        ind = scalar_diffusion(torch.ones(x.shape[0],1), t, method, {'L': par['L']})
        out = out*out_abs/(ind*out.norm(dim=-1, p=2, keepdim=True))
        
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