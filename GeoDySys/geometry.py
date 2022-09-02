#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import scipy.sparse as sp

import torch_geometric.utils as PyGu
from torch_geometric.nn import knn_graph, radius_graph
from torch_sparse import SparseTensor
from torch_scatter import scatter_add
from cknn import cknneighbors_graph

from torch.nn.functional import normalize

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

from ptu_dijkstra import ptu_dijkstra

from GeoDySys import utils


# =============================================================================
# Sampling
# =============================================================================
def sample_2d(N=100, interval=[[-1,-1],[1,1]], method='uniform', seed=0):
    
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


def furthest_point_sampling(X, N=None, return_clusters=False):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    
    Parameters
    ----------
    D : ndarray (N, N) 
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list) 
        (permutation (N-length array of indices), 
        lambdas (N-length array of insertion radii))
    """
    
    D = pairwise_distances(X, metric='euclidean')
    N = D.shape[0] if N is None else N
    
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
        
    if not return_clusters:
        return perm, lambdas
    else:
        clusters = [[] for i in range(N)]
        D = D[perm]
        D[D==0] = D.max()
        D[:,perm] = D.max()
        for i in range(X.shape[0]-N):
            idx = np.unravel_index(D.argmin(), D.shape)
            clusters[idx[0]].append(idx[1])
            D[:,idx[1]] = D.max()
            
        return perm, lambdas, clusters
        

# =============================================================================
# Clustering
# =============================================================================
def cluster_and_embed(x, cluster_typ='kmeans', 
                      embed_typ='tsne', 
                      n_clusters=15, 
                      proximity_order=True, 
                      seed=0):
    """Cluster & embed"""
    
    x = x.detach().numpy()

    clusters = cluster(x, cluster_typ, n_clusters, seed)
        
    #reorder to give close clusters similar labels
    if proximity_order:
        clusters = relabel_by_proximity(clusters)
        
    emb = np.vstack([x, clusters['centroids']])
    emb = embed(emb, embed_typ)
    clusters['centroids'] = emb[-n_clusters:]
    emb = emb[:-n_clusters]       
        
    return emb, clusters


def cluster(x, cluster_typ='kmeans', n_clusters=15, seed=0):
    
    clusters = dict()
    if cluster_typ=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(x)
        clusters['n_clusters'] = n_clusters
        clusters['labels'] = kmeans.labels_
        clusters['centroids'] = kmeans.cluster_centers_
    else:
        NotImplementedError
        
    return clusters


def embed(x, embed_typ='tsne'):
    
    if embed_typ=='tsne': 
        if x.shape[1]>2:
            print('Performed t-SNE embedding on embedded results.')
            emb = TSNE(init='random',learning_rate='auto').fit_transform(x)
    elif embed_typ=='umap':
        NotImplementedError
    else:
        NotImplementedError
    
    return emb


def relabel_by_proximity(clusters):
    
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
    nvec : (nxnxdim) Matrix of neighbourhood vectors.

    """
    
    n, dim = pos.shape
    
    mask = torch.zeros([n,n],dtype=bool)
    mask[edge_index[0], edge_index[1]] = 1
    mask = mask.unsqueeze(2).repeat(1,1,dim)
    
    nvec = pos.repeat(n,1,1)
    nvec = nvec - nvec.swapaxes(0,1) #nvec[i,j] = xj - xi 
    nvec[~mask] = 0
    
    return nvec


def gradient_op(nvec):
    """Gradient operator

    Parameters
    ----------
    nvec : (nxnxdim) Matrix of neighbourhood vectors.

    Returns
    -------
    G : (nxn) Gradient operator matrix.

    """
    
    G = torch.zeros_like(nvec)
    for i, g_ in enumerate(nvec):
        neigh_ind = torch.where(g_[:,0]!=0)[0]
        g_ = g_[neigh_ind]
        b = torch.column_stack([-1.*torch.ones((len(neigh_ind),1)),
                                torch.eye(len(neigh_ind))])
        grad = torch.linalg.lstsq(g_, b).solution
        G[i,i,:] = grad[:,[0]].T
        G[i,neigh_ind,:] = grad[:,1:].T
            
    return [G[...,i] for i in range(G.shape[-1])]


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


def DD(pos, edge_index, gauges, order=1, include_identity=False):
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
    
    nvec = neighbour_vectors(pos, edge_index) #(nxnxdim)
    F = project_gauge_to_neighbours(nvec, gauges)

    if include_identity:
        K = [torch.eye(F[0].shape[0])]
    else:
        K = []
        
    for _F in F:
        Fhat = normalize(_F, dim=-1, p=1)
        K.append(Fhat - torch.diag(torch.sum(Fhat, dim=1)))
        
    #derivative orders
    if order>1:
        n = len(K)
        K0 = K
        for i in range(order-1):
            Ki = [K0[j]*K0[k] for j in range(n) for k in range(n)]
            K += Ki
    
    return K


# def DA(data, gauges):
#     """
#     Directional average kernel from Beaini et al. 2021.

#     Parameters
#     ----------
#     data : pytorch geometric data object containing .pos and .edge_index
#     gauges : (n,dim,dim) matric of orthogonal unit vectors for each node

#     Returns
#     -------
#     K : list of (nxn) Anisotropic kernels.

#     """
    
#     nvec = neighbour_vectors(data.pos, data.edge_index) #(nxnxdim)
#     F = project_gauge_to_neighbours(nvec, gauges)

#     K = []
#     for _F in F:
#         Fhat = normalize(_F, dim=-1, p=1)
#         K.append(torch.abs(Fhat))
        
#     return K


def compute_gauges(data, local=False, n_geodesic_nb=10):
    """
    Compute gauges

    Parameters
    ----------
    data : pytorch geometric data object containing .pos and .edge_index
    local : bool, The default is False.
    n_geodesic_nb : int, The default is 10.

    Returns
    -------
    gauges : (n,dim,dim) matric of orthogonal unit vectors for each node
    R : (n,n,dim,dim) connection matrices. If local=True.

    """
    
    n = data.pos.shape[0]
    dim = data.pos.shape[-1]
    
    if local:
        gauges, R = compute_tangent_bundle(
            data, 
            n_geodesic_nb=n_geodesic_nb,
            )
        return gauges, R
    else:      
        gauges = torch.eye(dim)
        gauges = gauges.repeat(n,1,1)      
        return gauges, None


def project_gauge_to_neighbours(nvec, gauges):
    """
    Project the gauge vectors to local edge vectors.
    
    Parameters
    ----------
    nvec : (nxnxdim) Matrix of neighbourhood vectors.
    local_gauge : dimxnxdim torch tensor, if None, global gauge is generated

    Returns
    -------
    F : list of (nxn) torch tensors of projected components
    
    """
            
    gauges = gauges.swapaxes(0,1) #(nxdimxdim) -> (dimxnxdim)
    F = [(nvec*g).sum(-1) for g in gauges] #dot product in last dimension
        
    return F


def adjacency_matrix(edge_index, size=None, value=None):
    """
    Adjacency matrix as torch_sparse tensor

    Parameters
    ----------
    edge_index : (2x|E|) Matrix of edge indices
    size : pair (rows,cols) giving the size of the matrix. 
        The default is the largest node of the edge_index.
    value : list of weights. The default is unit values.

    Returns
    -------
    adj : TYPE
        DESCRIPTION.

    """    
    if value is not None:
        value = value[edge_index[0], edge_index[1]]
    if size is None:
        size = (edge_index.max()+1, edge_index.max()+1)
        
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                       value=value,
                       sparse_sizes=(size[0], size[1]))
    
    return adj


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
    
    return edge_index, edge_weight


def compute_laplacian(data, normalization="rw"):
    
    L = PyGu.get_laplacian(data.edge_index,
                           edge_weight = data.edge_weight,
                           normalization = normalization)
    L = PyGu.to_scipy_sparse_matrix(L[0], edge_attr=L[1])
    
    return L


def compute_connection_laplacian(data, R=None, normalization='rw'):
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
    L = sp.kron(L, sp.csr_matrix(torch.ones([dim,dim])))
    L = utils.np2torch(L.toarray())
    
    if R is None:
        R = torch.ones([n*dim, n*dim])
    else:
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
        
    return sp.csr_matrix(Lc)


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
    A_eigsh = (A + sp.identity(A.shape[0])*eps).tocsc()
    failcount = 0
    while True:
        try:
            evals, evecs = sp.linalg.eigsh(A_eigsh, k=k)
            
            # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
            evals = np.clip(evals, a_min=0., a_max=float('inf'))

            break
        except Exception as e:
            print(e)
            if(failcount > 3):
                raise ValueError("failed to compute eigendecomp")
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            A_eigsh = A_eigsh + sp.identity(A.shape[0]) * (eps * 10**failcount)
    
    return utils.np2torch(evals), utils.np2torch(evecs)


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

    _, _, tangents, R = ptu_dijkstra(X, A, X.shape[1], n_geodesic_nb, return_predecessors)
    
    return utils.np2torch(tangents), utils.np2torch(R)
    
    
def compute_connections(gauges, A, dim_man=None):
    
    assert len(gauges.shape)==3, 'Gauges need to be a nxdxk matrix.'
    
    n, d, k = gauges.shape
    
    assert dim_man <= d, 'Manifold dimension should be no more than that of \
                          the embedding space!'
    
    if dim_man is not None:
        if dim_man == d: #the manifold is the whole space
            return None
        else:
            gauges = gauges[:,:,dim_man:]    
    
    R = np.zeros([n,n,d,d])
    for i in range(n):
        for j in range(n):
            if A[i,j] != 0:
                R[i,j,...] = procrustes(gauges[i,:], gauges[j,:])

    return R



def compute_diffusion(x, t, L, method='matrix_exp'):
    if method == 'matrix_exp':
        return sp.linalg.expm_multiply(-t*L, x)
    
    
def procrustes(X, Y):
    """

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    Outputs
    ------------
    T

    """

    # optimum rotation matrix of Y
    A = np.dot(X.T, Y)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # does the current solution use a reflection?
    have_reflection = np.linalg.det(T) < 0

    # if that's not what was specified, force another reflection
    if have_reflection:
        V[:,-1] *= -1
        s[-1] *= -1
        T = np.dot(V, U.T)
       
    return T