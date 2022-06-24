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


# =============================================================================
# Sampling
# =============================================================================
def sample_2d(n=100, interval=[[-1,-1],[1,1]], method='uniform', seed=0):
    
    if method=='uniform':
        x = np.linspace(interval[0][0], interval[1][0], int(np.sqrt(n)))
        y = np.linspace(interval[0][1], interval[1][1], int(np.sqrt(n)))
        x = np.stack([x,y],axis=1)
        
    elif method=='random':
        np.random.seed(seed)
        x = np.random.uniform((interval[0][0], interval[0][1]), 
                              (interval[1][0], interval[1][1]), 
                              (n,2))         
    return x


def furthest_point_sampling(X, N=None):
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
        
    return perm, lambdas


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
# Functions for plotting on sphere
# =============================================================================
def grid_sphere(b, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    theta, phi = np.meshgrid(*linspace(b, grid_type), indexing='ij')
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def linspace(b, grid_type='Driscoll-Healy'):
    if grid_type == 'Driscoll-Healy':
        beta = np.arange(2 * b) * np.pi / (2. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'SOFT':
        beta = np.pi * (2 * np.arange(2 * b) + 1) / (4. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'Clenshaw-Curtis':
        # beta = np.arange(2 * b + 1) * np.pi / (2 * b)
        # alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
        # Must use np.linspace to prevent numerical errors that cause beta > pi
        beta = np.linspace(0, np.pi, 2 * b + 1)
        alpha = np.linspace(0, 2 * np.pi, 2 * b + 2, endpoint=False)
    elif grid_type == 'equidistribution':
        raise NotImplementedError('Not implemented yet; see Fast evaluation of quadrature formulae on the sphere.')
    else:
        raise ValueError('Unknown grid_type:' + grid_type)
        
    return beta, alpha


def project_2d_on_sphere(signal, grid, projection_origin=None):
    ''' '''
    NORTHPOLE_EPSILON = 1e-3
    
    if projection_origin is None:
        projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    sample = sample_bilinear(signal, rx, ry)

    # ensure that only south hemisphere gets projected
    sample *= (grid[2] <= 1).astype(np.float64)

    # rescale signal to [0,1]
    sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
    sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

    sample = (sample - sample_min) / (sample_max - sample_min)
    sample *= 255
    sample = sample.astype(np.uint8)

    return sample


def project_sphere_on_xy_plane(grid, projection_origin):
    ''' returns xy coordinates on the plane
    obtained from projecting each point of
    the spherical grid along the ray from
    the projection origin through the sphere '''

    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    # ensure that plane projection
    # ends up on southern hemisphere
    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry


def sample_bilinear(signal, rx, ry):
    ''' '''

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # discretize sample position
    ix = rx.astype(int)
    iy = ry.astype(int)

    # obtain four sample coordinates
    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # linear interpolation in x-direction
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def sample_within_bounds(signal, x, y, bounds):
    ''' '''
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
        
    return sample


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    return (np.outer(V, V) - np.eye(3)).dot(R)


def rotate_grid(rot, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    
    return x_r, y_r, z_r


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


# def build_tangent_frames(verts):
#     #assigns arbitrary orthogonal tangent coordinates to vertices

#     V = verts.shape[0]

#     vert_normals = vertex_normals(verts)  # (V,3)

#     # = find an orthogonal basis
#     basis_cand1 = torch.tensor([1, 0, 0]).expand(V, -1)
#     basis_cand2 = torch.tensor([0, 1, 0]).expand(V, -1)
    
#     basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1))
#                           < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
#     basisX = project_to_tangent(basisX, vert_normals)
#     basisX = normalize(basisX)
#     basisY = cross(vert_normals, basisX)
#     frames = torch.stack((basisX, basisY, vert_normals), dim=-2)
    
#     if torch.any(torch.isnan(frames)):
#         raise ValueError("NaN coordinate frame! Must be very degenerate")

#     return frames


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