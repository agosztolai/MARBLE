#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as sla
import scipy

from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

from .utils import np2torch

from ptu_dijkstra import ptu_dijkstra


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
def cluster(emb, typ='kmeans', n_clusters=15, reorder=True, tsne_embed=True, seed=0):
    """Cluster embedding"""
    
    emb = emb.detach().numpy()
    
    clusters = dict()
    if typ=='kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(emb)
        clusters['n_clusters'] = n_clusters
        clusters['labels'] = kmeans.labels_
        clusters['centroids'] = kmeans.cluster_centers_
    else:
        NotImplementedError
        
    #reorder to give close clusters similar labels
    if reorder:
        pd = pairwise_distances(clusters['centroids'], metric='euclidean')
        pd += np.max(pd)*np.eye(n_clusters)
        mapping = {}
        id_old = 0
        for i in range(n_clusters):
            id_new = np.argmin(pd[id_old,:])
            while id_new in mapping.keys():
                pd[id_old,id_new] += np.max(pd)
                id_new = np.argmin(pd[id_old,:])
            mapping[id_new] = i
            id_old = id_new
            
        l = clusters['labels']
        clusters['labels'] = np.array([mapping[l[i]] for i,_ in enumerate(l)])
        clusters['centroids'] = clusters['centroids'][list(mapping.keys())]
        
    if tsne_embed:
        n_emb = emb.shape[0]
        emb = np.vstack([emb, clusters['centroids']])
        if emb.shape[1]>2:
            print('Performed t-SNE embedding on embedded results.')
            emb = TSNE(init='random',learning_rate='auto').fit_transform(emb)
            
        clusters['centroids'] = emb[n_emb:]
        emb = emb[:n_emb]       
        
        
    return emb, clusters


# =============================================================================
# Discrete differential operators
# =============================================================================
def compute_laplacian(data, k_eig=2, eps=1e-8):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.
    Arguments:
      - k_eig: number of eigenvectors to use
    Returns:
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian 
      - grad_mat: (VxVxdim) sparse matrix which gives the gradient in the local basis at the vertex
    """

    L = get_laplacian(data.edge_index, normalization="rw")
    L = to_scipy_sparse_matrix(L[0], edge_attr=L[1])
    
    # Compute the eigenbasis
    L_eigsh = (L + scipy.sparse.identity(L.shape[0])*eps).tocsc()
    failcount = 0
    while True:
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=k_eig)
            
            # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
            evals = np.clip(evals, a_min=0., a_max=float('inf'))

            break
        except Exception as e:
            print(e)
            if(failcount > 3):
                raise ValueError("failed to compute eigendecomp")
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10**failcount)


    # == Build gradient matrices
    # frames = build_tangent_frames(verts)
    # grad_mat = build_grad(verts, frames)
    
    return [np2torch(L.toarray()), None, np2torch(evals), np2torch(evecs)]


def compute_tangent_frames(data, n_geodesic_neighbours=10):

    X = data.pos.numpy().astype(np.float64)
    A = to_scipy_sparse_matrix(data.edge_index).tocsr()

    tangents = ptu_dijkstra(X, A, 2, n_geodesic_neighbours, return_predecessors=False)
    
    return tangents.swapaxes(0,1)


# def vertex_normals(verts, n_nb=30):
    
#     _, neigh_inds = find_knn(verts, verts, n_nb, omit_diagonal=True, method='cpu_kd')
#     neigh_points = verts[neigh_inds,:]
#     neigh_vecs = neigh_points - verts[:,np.newaxis,:]
    
#     (u, s, vh) = np.linalg.svd(neigh_vecs, full_matrices=False)
#     normal = vh[:,2,:]
#     normal /= np.linalg.norm(normal,axis=-1, keepdims=True)
        
#     if torch.any(torch.isnan(normal)): raise ValueError("NaN normals :(")

#     return normal


# def build_grad_point_cloud(verts, frames, n_nb=30):

#     _, neigh_inds = find_knn(verts, verts, n_nb, omit_diagonal=True, method='cpu_kd')

#     edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_nb)
#     edges = np.stack((edge_inds_from, neigh_inds.flatten()))
#     edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)#this is the F in Beaini (?)
    
#     return build_grad(verts, torch.tensor(edges), edge_tangent_vecs)


# def edge_tangent_vectors(verts, frames, edges):
#     edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
#     basisX = frames[edges[0, :], 0, :]
#     basisY = frames[edges[0, :], 1, :]

#     compX = edge_vecs.dot(basisX)
#     compY = edge_vecs.dot(basisY)
#     edge_tangent = torch.stack((compX, compY), dim=-1)

#     return edge_tangent


# def project_to_tangent(vecs, unit_normals):
#     dots = vecs.dot(unit_normals)
#     return vecs - unit_normals * dots.unsqueeze(-1)