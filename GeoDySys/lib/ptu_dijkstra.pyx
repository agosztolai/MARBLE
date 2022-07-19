"""
The Fibbonacci Heap data structure and several components related to Dijkstra's
standard algorithm were adopted from `scipy.sparse.csgraph._shortest_path.pyx`,
authored and copywrited by Jake Vanderplas  -- <vanderplas@astro.washington.edu>
under BSD in 2011.

Author of all additional code pertaining to the PTU Dijkstra Algorithm is
Max Budninskiy (C). License: modified (3-clause) BSD, 2020.
"""
import warnings

import numpy as np
from scipy.sparse.csgraph._validation import validate_graph

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
cimport cython
cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t


def ptu_dijkstra(X,
                 csgraph,
                 d,
                 K,
                 return_predecessors=False):
    """
    Parallel Transport Dijkstra algorithm using Fibonacci Heaps

    Parameters
    ----------
    X: numpy matrix
        (N, D) matrix of N input data points in D dimensional space sampling a
        lower dimensional manifold S
    csgraph : sparse matrix
        Distance weighted proximity graph of pointset X
    d : int
        Dimension of the manifold S
    K : int
        Number of points to include in geodesic neighborhoods. Geodesic
        neighborhood of a point x of size K is K nearest neighbors to x in
        the proximity graph. Notice it's different than simple K nearest
        neighbors in ambient D dimensional space. Geodesic neighborhood of
        point x is used to compute local tangent space to the data manifold at
        x.
    return_predecessors : bool, optional
        If True, return the size (N, N) predecesor matrix

    Returns
    -------
    dist_matrix : numpy matrix
        The (N, N) matrix of geodesic distances between points of X, that sample
        a d-dimensional manifold S. These distances are computed using parallel
        transport unfolding algorithm.
    predecessors : numpy matrix
        Returned only if return_predecessors == True.
        The (N, N) matrix of predecessors, which can be used to reconstruct
        the shortest paths.  Row i of the predecessor matrix contains
        information on the shortest paths from point i: each entry
        predecessors[i, j] gives the index of the previous node in the
        path from point i to point j.  If no path exists between point
        i and j, then predecessors[i, j] = -1

    Notes
    -----
    The input csgraph is symmetrized first.
    """
    N = X.shape[0]
    D = X.shape[1]
    cdef ITYPE_t N_t = N
    cdef ITYPE_t K_t = K
    cdef ITYPE_t D_t = D
    cdef ITYPE_t d_t = d

    if K >= N:
        raise ValueError(
            "Geodesic neighborhood size must be less than the "
            "total number of samples"
        )
    if K < d:
        raise ValueError(
            "Geodesic neighborhood size must be larger or equal to the "
            "embedding dimension"
        )
    if D < d:
        raise ValueError(
            "Embedding dimension must be less or equal to the ambient "
            "dimension of input data"
        )

    csgraph = validate_graph(csgraph, directed=True, dtype=DTYPE,
                             dense_output=False)

    if np.any(csgraph.data < 0):
        warnings.warn("Graph has negative weights: \
                      negative distances are not allowed.")

    # initialize ptu distances, predecessors, and tangent spaces
    ptu_dists = np.zeros((N, N), dtype=DTYPE)
    ptu_dists.fill(np.inf)
    predecessors = np.empty((N, N), dtype=ITYPE)
    predecessors.fill(-1)
    tangents = np.empty((N, D, d), dtype=DTYPE)
    R = np.empty(shape=[N, N, d, d], dtype=DTYPE)


    # symmetrize the graph
    csgraphT = csgraph.T.tocsr()
    symmetrized_graph = csgraph.maximum(csgraphT)
    graph_data = symmetrized_graph.data
    graph_indices = symmetrized_graph.indices
    graph_indptr = symmetrized_graph.indptr

    tangents_status = _geodesic_neigborhood_tangents(
            X,
            graph_data,
            graph_indices,
            graph_indptr,
            tangents,
            N_t,
            K_t,
            D_t,
            d_t
    )
    if tangents_status == -1:
        raise RuntimeError(
            'Local tangent space approximation failed, at least one geodesic '
            'neighborhood does not span d-dimensional space'
        )
    elif tangents_status == 1:
        _parallel_transport_dijkstra(
            X,
            graph_data,
            graph_indices,
            graph_indptr,
            ptu_dists,
            tangents,
            predecessors,
            R,
            N_t,
            D_t,
            d_t
        )
        if return_predecessors:
            return ptu_dists, predecessors, tangents, R
        return ptu_dists, None, tangents, R
    else:
        raise RuntimeError(
            'Local tangent space approximation failed'
        )


@cython.boundscheck(False)
cdef _dijkstra_scan_heap(FibonacciHeap *heap,
                         FibonacciNode *v,
                         FibonacciNode *nodes,
                         const double[:] csr_weights,
                         const int[:] csr_indices,
                         const int[:] csr_indptr,
                         int[:, :] pred,
                         int return_pred,
                         int i):
    cdef:
        unsigned int k, j_source, j_current
        ITYPE_t j
        DTYPE_t next_val
        FibonacciNode *current_node

    for j in range(csr_indptr[v.index], csr_indptr[v.index + 1]):
        j_current = csr_indices[j]
        current_node = &nodes[j_current]
        if current_node.state != SCANNED:
            next_val = v.val + csr_weights[j]
            if current_node.state == NOT_IN_HEAP:
                current_node.state = IN_HEAP
                current_node.val = next_val
                insert_node(heap, current_node)
                if return_pred:
                    pred[i, j_current] = v.index
            elif current_node.val > next_val:
                decrease_val(heap, current_node,
                             next_val)
                if return_pred:
                    pred[i, j_current] = v.index


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _parallel_transport_dijkstra(
            double[:, :] X,
            double[:] csr_weights,
            int[:] csr_indices,
            int[:] csr_indptr,
            double[:, :] ptu_dists,
            double[:, :, :] tangents,
            int[:, :] pred,
            double[:, :, :, :] R,
            int N,
            int D,
            int d
            ):
    """
    Performs parallel transport Dijkstra pairwise distance estimation.

    Parameters:
    X: matrix
        The (N, D) matrix of N input data points in D dimensional space sampling
        a lower dimensional manifold S.
    csr_weights: array
        Values of sparse csr distance weighted adjacency matrix
        representing proximity graph of pointset X.
    csr_indices: array (int)
        Indices of sparce csr proximity graph matrix.
    csr_indptr: array (int)
        Index pointers of sparce csr proximity graph matrix.
    ptu_dists: matrix
        [Output] Matrix of (N, N) pairwise geodesic distances between points of
        pointset X (geodesic with repect to manifold S).
    tangents: 3 dimensional tensor
        Collection of N local tangent space bases of size (D, d).
    pred: matrix
        [Output] [N, N] Matrix of Dijkstra algorithm predecessors.
    N: int
        Number of points in dataset X.
    D: int
        Ambient dimension of pointset X.
    d: int
        Dimension of manifold S that is sampled by X.
    """
    cdef:
        int i, k, l, p, q, j, pred_j, k_current
        int info, lwork = 6*d
        double edge_norm, temp, next_val, path_norm, projected_edge_norm

        np.ndarray[DTYPE_t, ndim=3] iTangent = np.empty(shape=[N, D, d], dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] jTangent = np.empty(shape=[D, d], dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] iTangentPred = np.empty(shape=[D, d], dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] graph_dists = np.empty(shape=[N, N], dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] iUnfolding = np.empty(shape=[N, d], dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] projected_edge = np.zeros(shape=d, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] Work = np.empty(shape=[lwork], dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] TtT = np.empty(shape=[d, d], dtype=DTYPE, order='F')
        np.ndarray[DTYPE_t, ndim=2] U = np.empty(shape=[d, d], dtype=DTYPE, order='F')
        np.ndarray[DTYPE_t, ndim=2] VT = np.empty(shape=[d, d], dtype=DTYPE, order='F')
        np.ndarray[DTYPE_t, ndim=1] S = np.empty(shape=d, dtype=DTYPE)

        FibonacciHeap heap
        FibonacciNode *v
        FibonacciNode *nodes = <FibonacciNode*> malloc(N * sizeof(FibonacciNode))
        FibonacciNode *current_node

    if nodes == NULL:
        raise MemoryError("Failed to allocate memory in _dijkstra_undirected")

    for i in range(N):

        # initialize nodes
        for k in range(N):
            initialize_node(&nodes[k], k)

        # initialize tangent at i that every other tangent will be aligned to
        for p in range(0, D):
            for q in range(0, d):
                iTangent[i, p, q] = tangents[i, p, q]

        # initialize the matrix of geodesics from i to all other points
        # unfolded into d-dimensional tangent space centered at i
        for p in range(0, N):
            for q in range(0, d):
                iUnfolding[p, q] = 0

        # initialize parallel transport distances
        ptu_dists[i, i] = 0

        # initialize Dijkstra graph distances
        graph_dists[i, i] = 0

        # initialize predecessors
        pred[i, i] = -1

        # initialize Fibonacci heap
        heap.min_node = NULL
        insert_node(&heap, &nodes[i])

        # start Dijkstra algorithm
        while heap.min_node:
            v = remove_min(&heap)
            v.state = SCANNED
            j = v.index
            pred_j = pred[i, j]
            graph_dists[i, j] = v.val

            if pred_j != -1:
                # jTangent is tangent space to node j unaligned to tangent space
                # at node i. iTangentPred is tangent space to node preceding j
                # aligned to tangent space at node i.
                for p in range(0, D):
                    for q in range(0, d):
                        jTangent[p, q] = tangents[j, p, q]
                        iTangentPred[p, q] = iTangent[pred_j, p, q]

                for p in range(d):
                    projected_edge[p] = 0

                # projected_edge = (X[j] - X[pred_j])^T * iTangentPred
                # edge_norm = ||X[j] - X[pred_j]||_2
                edge_norm = 0
                for p in range(D):
                    temp = X[j, p] - X[pred_j, p]
                    edge_norm += temp * temp
                    for q in range(d):
                        projected_edge[q] += temp * iTangentPred[p, q]
                edge_norm = sqrt(edge_norm)

                # projected_edge_norm = ||projected_edge||_2
                projected_edge_norm = 0
                for q in range(0, d):
                    projected_edge_norm += projected_edge[q] * projected_edge[q]
                projected_edge_norm = sqrt(projected_edge_norm)

                # tweak the length of projected edge to eliminate shrinking
                # and add it to the unwrapped path from i to predecessor of j.
                path_norm = 0
                for q in range(0, d):
                    projected_edge[q] = (
                        edge_norm / projected_edge_norm * projected_edge[q]
                    )
                    projected_edge[q] += iUnfolding[pred_j, q]
                    iUnfolding[j, q] = projected_edge[q]
                    path_norm += projected_edge[q] * projected_edge[q]
                path_norm = sqrt(path_norm)

                # eucledian length of the unfolded path from i to j is the
                # parallel transport approximation of geodesic distance between
                # points i and j
                ptu_dists[i, j] = path_norm

                # TtT = jTangent^T iTangentPred
                for q in range(0, d):
                    for k in range(0, d):
                        temp = 0
                        for p in range(0, D):
                            temp += jTangent[p, k] * iTangentPred[p, q]
                        TtT[k, q] = temp

                # U, S, VT = SVD(TtT)
                # see LAPACK docs for details
                cython_lapack.dgesvd(
                    'A',
                    'A',
                    &d,
                    &d,
                    &TtT[0, 0],
                    &d,
                    &S[0],
                    &U[0, 0],
                    &d,
                    &VT[0, 0],
                    &d,
                    &Work[0],
                    &lwork,
                    &info
                )

                # iTangent[j, :, :] = jTangent * U * VT
                # this is a tangent space at node j that is aligned to tangent
                # space at node i
                for p in range(D):
                    for q in range(d):
                        temp = 0
                        for k in range(0, d):
                            for l in range(0, d):
                                temp += jTangent[p, k] * U[k, l] * VT[l, q]
                        iTangent[j, p, q] = temp
                      
                for q in range(d):
                    for k in range(0, d):
                        temp = 0
                        for l in range(0, d):
                            temp += U[k, l] * VT[l, q]
                        R[i,j,k,q] = temp

            # Standard Dijkstra: process neighbors of j
            for k in range(csr_indptr[j], csr_indptr[j + 1]):
                k_current = csr_indices[k]
                current_node = &nodes[k_current]
                if current_node.state != SCANNED:
                    next_val = v.val + csr_weights[k]
                    if current_node.state == NOT_IN_HEAP:
                        current_node.state = IN_HEAP
                        current_node.val = next_val
                        insert_node(&heap, current_node)
                        pred[i, k_current] = v.index
                    elif current_node.val > next_val:
                        decrease_val(&heap, current_node,
                                     next_val)
                        pred[i, k_current] = v.index

            # this function can be used instead, but it has slighly worse timing
            # _dijkstra_scan_heap(&heap, v, nodes,
            #                     csr_weights, csr_indices, csr_indptr,
            #                     pred, return_pred, i)

    free(nodes)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _geodesic_neigborhood_tangents(
            double[:, :] X,
            double[:] csr_weights,
            int[:] csr_indices,
            int[:] csr_indptr,
            double[:, :, :] tangents,
            int N,
            int K,
            int D,
            int d,
            ):
    """
    Computes a tangent space for every input point using geodesic neighborhoods.

    Parameters:
    X: matrix
        The (N, D) matrix of N input data points in D dimensional space sampling
        a lower dimensional manifold S.
    csr_weights: array
        Values of sparse csr distance weighted adjacency matrix
        representing proximity graph of pointset X.
    csr_indices: array (int)
        Indices of sparce csr proximity graph matrix.
    csr_indptr: array (int)
        Index pointers of sparce csr proximity graph matrix.
    tangets: 3 dimensional tensor
        [Output] Collection of N local tangent space bases of size (D, d).
    N: int
        Number of points in dataset X.
    K: int
        Number of points used to define a gedesic neighborhood.
    D: int
        Ambient dimension of pointset X.
    d: int
        Dimension of manifold S that is sampled by X.
    """
    cdef:
        unsigned int i, j, k, l, p, q, scanned_cntr, k_current
        int return_pred = 0, Kp1 = K + 1, mn = min(D, Kp1)
        int info, lwork = max(3*min(D, Kp1) + max(D, Kp1), 5*min(D, Kp1))
        double mean, next_val

        np.ndarray[DTYPE_t, ndim=2] geoNbh = np.empty(shape=[D, Kp1], dtype=DTYPE, order='F')
        np.ndarray[DTYPE_t, ndim=2] U = np.empty(shape=(D, mn), dtype=DTYPE, order='F')
        np.ndarray[DTYPE_t, ndim=2] VT = np.empty(shape=(mn, Kp1), dtype=DTYPE, order='F')
        np.ndarray[DTYPE_t, ndim=1] S = np.empty(shape=mn, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] Work = np.empty(shape=[lwork], dtype=DTYPE)
        np.ndarray[ITYPE_t, ndim=1] geoNbh_indices = np.empty(shape=Kp1, dtype=ITYPE)

        FibonacciHeap heap
        FibonacciNode *v
        FibonacciNode *nodes = <FibonacciNode*> malloc(N *
                                                       sizeof(FibonacciNode))
        FibonacciNode *current_node

    if nodes == NULL:
        raise MemoryError("Failed to allocate memory in _geodesic_neigborhood_tangents")

    for i in range(N):

        # initialize nodes for Dijkstra
        for k in range(N):
            initialize_node(&nodes[k], k)

        # insert node i into heap
        heap.min_node = NULL
        insert_node(&heap, &nodes[i])

        # counter of processed points closest to i
        scanned_cntr = 0

        # perform standard Dijkstra until K closest points are discovered
        # keep track of the indices of these K points
        while (heap.min_node) and (scanned_cntr <= K):
            v = remove_min(&heap)
            v.state = SCANNED
            j = v.index
            geoNbh_indices[scanned_cntr] = j
            scanned_cntr += 1
            if scanned_cntr <= K:
                for k in range(csr_indptr[j], csr_indptr[j + 1]):
                    k_current = csr_indices[k]
                    current_node = &nodes[k_current]
                    if current_node.state != SCANNED:
                        next_val = v.val + csr_weights[k]
                        if current_node.state == NOT_IN_HEAP:
                            current_node.state = IN_HEAP
                            current_node.val = next_val
                            insert_node(&heap, current_node)
                        elif current_node.val > next_val:
                            decrease_val(&heap, current_node,
                                         next_val)

        # construct and center geodesic neighborhood from indices
        for p in range(D):
            mean = 0
            for q in range(Kp1):
                geoNbh[p, q] = X[geoNbh_indices[q], p]
                mean += geoNbh[p, q]
            mean = mean / Kp1
            for q in range(Kp1):
                geoNbh[p, q] -= mean

        # perform SVD of the geodesic neighborhood points
        # see LAPACK docs for details
        cython_lapack.dgesvd(
            'S',
            'N',
            &D,
            &Kp1,
            &geoNbh[0, 0],
            &D,
            &S[0],
            &U[0, 0],
            &D,
            &VT[0, 0],
            &mn,
            &Work[0],
            &lwork,
            &info
        )

        # d left singular vectors form a basis for tangent space at point i
        for q in range(d):
            if S[q] < 1e-10:
                return -1
            for p in range(D):
                tangents[i, p, q] = U[p, q]

    free(nodes)
    return 1

######################################################################
# FibonacciNode structure
#  This structure and the operations on it are the nodes of the
#  Fibonacci heap.
#
cdef enum FibonacciState:
    SCANNED
    NOT_IN_HEAP
    IN_HEAP


cdef struct FibonacciNode:
    unsigned int index
    unsigned int rank
    unsigned int source
    FibonacciState state
    DTYPE_t val
    FibonacciNode* parent
    FibonacciNode* left_sibling
    FibonacciNode* right_sibling
    FibonacciNode* children

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void initialize_node(FibonacciNode* node,
                          unsigned int index,
                          DTYPE_t val=0):
    # Assumptions: - node is a valid pointer
    #              - node is not currently part of a heap
    node.index = index
    node.source = -9999
    node.val = val
    node.rank = 0
    node.state = NOT_IN_HEAP

    node.parent = NULL
    node.left_sibling = NULL
    node.right_sibling = NULL
    node.children = NULL

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef FibonacciNode* rightmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.right_sibling):
        temp = temp.right_sibling
    return temp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef FibonacciNode* leftmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.left_sibling):
        temp = temp.left_sibling
    return temp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void add_child(FibonacciNode* node, FibonacciNode* new_child):
    # Assumptions: - node is a valid pointer
    #              - new_child is a valid pointer
    #              - new_child is not the sibling or child of another node
    new_child.parent = node

    if node.children:
        add_sibling(node.children, new_child)
    else:

        node.children = new_child
        new_child.right_sibling = NULL
        new_child.left_sibling = NULL
        node.rank = 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void add_sibling(FibonacciNode* node, FibonacciNode* new_sibling):
    # Assumptions: - node is a valid pointer
    #              - new_sibling is a valid pointer
    #              - new_sibling is not the child or sibling of another node
    cdef FibonacciNode* temp = rightmost_sibling(node)
    temp.right_sibling = new_sibling
    new_sibling.left_sibling = temp
    new_sibling.right_sibling = NULL
    new_sibling.parent = node.parent
    if new_sibling.parent:
        new_sibling.parent.rank += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void remove(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    if node.parent:
        node.parent.rank -= 1
        if node.left_sibling:
            node.parent.children = node.left_sibling
        elif node.right_sibling:
            node.parent.children = node.right_sibling
        else:
            node.parent.children = NULL

    if node.left_sibling:
        node.left_sibling.right_sibling = node.right_sibling
    if node.right_sibling:
        node.right_sibling.left_sibling = node.left_sibling

    node.left_sibling = NULL
    node.right_sibling = NULL
    node.parent = NULL


######################################################################
# FibonacciHeap structure
#  This structure and operations on it use the FibonacciNode
#  routines to implement a Fibonacci heap

ctypedef FibonacciNode* pFibonacciNode


cdef struct FibonacciHeap:
    FibonacciNode* min_node
    pFibonacciNode[100] roots_by_rank  # maximum number of nodes is ~2^100.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void insert_node(FibonacciHeap* heap,
                      FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    if heap.min_node:
        add_sibling(heap.min_node, node)
        if node.val < heap.min_node.val:
            heap.min_node = node
    else:
        heap.min_node = node

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void decrease_val(FibonacciHeap* heap,
                       FibonacciNode* node,
                       DTYPE_t newval):
    # Assumptions: - heap is a valid pointer
    #              - newval <= node.val
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    #              - node is in the heap
    node.val = newval
    if node.parent and (node.parent.val >= newval):
        remove(node)
        insert_node(heap, node)
    elif heap.min_node.val > node.val:
        heap.min_node = node

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void link(FibonacciHeap* heap, FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is already within heap

    cdef FibonacciNode *linknode
    cdef FibonacciNode *parent
    cdef FibonacciNode *child

    if heap.roots_by_rank[node.rank] == NULL:
        heap.roots_by_rank[node.rank] = node
    else:
        linknode = heap.roots_by_rank[node.rank]
        heap.roots_by_rank[node.rank] = NULL

        if node.val < linknode.val or node == heap.min_node:
            remove(linknode)
            add_child(node, linknode)
            link(heap, node)
        else:
            remove(node)
            add_child(linknode, node)
            link(heap, linknode)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef FibonacciNode* remove_min(FibonacciHeap* heap):
    # Assumptions: - heap is a valid pointer
    #              - heap.min_node is a valid pointer
    cdef:
        FibonacciNode *temp
        FibonacciNode *temp_right
        FibonacciNode *out
        unsigned int i

    # make all min_node children into root nodes
    if heap.min_node.children:
        temp = leftmost_sibling(heap.min_node.children)
        temp_right = NULL

        while temp:
            temp_right = temp.right_sibling
            remove(temp)
            add_sibling(heap.min_node, temp)
            temp = temp_right

        heap.min_node.children = NULL

    # choose a root node other than min_node
    temp = leftmost_sibling(heap.min_node)
    if temp == heap.min_node:
        if heap.min_node.right_sibling:
            temp = heap.min_node.right_sibling
        else:
            out = heap.min_node
            heap.min_node = NULL
            return out

    # remove min_node, and point heap to the new min
    out = heap.min_node
    remove(heap.min_node)
    heap.min_node = temp

    # re-link the heap
    for i in range(100):
        heap.roots_by_rank[i] = NULL

    while temp:
        if temp.val < heap.min_node.val:
            heap.min_node = temp
        temp_right = temp.right_sibling
        link(heap, temp)
        temp = temp_right

    return out


######################################################################
# Debugging: Functions for printing the Fibonacci heap
#
#cdef void print_node(FibonacciNode* node, int level=0):
#    print '%s(%i,%i) %i' % (level*'   ', node.index, node.val, node.rank)
#    if node.children:
#        print_node(leftmost_sibling(node.children), level+1)
#    if node.right_sibling:
#        print_node(node.right_sibling, level)
#
#
#cdef void print_heap(FibonacciHeap* heap):
#    print "---------------------------------"
#    print "min node: (%i, %i)" % (heap.min_node.index, heap.min_node.val)
#    if heap.min_node:
#        print_node(leftmost_sibling(heap.min_node))
#    else:
#        print "[empty heap]"