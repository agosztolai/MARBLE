"""Module imported and adapted from https://github.com/chlorochrule/cknn."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def cknneighbors_graph(
    X,
    n_neighbors,
    delta=1.0,
    metric="euclidean",
    t="inf",
    include_self=False,
    is_sparse=True,
    return_instance=False,
):
    """Main function to call, see CkNearestNeighbors for the doc."""
    cknn = CkNearestNeighbors(
        n_neighbors=n_neighbors,
        delta=delta,
        metric=metric,
        t=t,
        include_self=include_self,
        is_sparse=is_sparse,
    )
    cknn.cknneighbors_graph(X)

    if return_instance:
        return cknn
    return cknn.ckng


class CkNearestNeighbors(object):
    """This object provides the all logic of CkNN.

    Args:
        n_neighbors: int, optional, default=5
            Number of neighbors to estimate the density around the point.
            It appeared as a parameter `k` in the paper.

        delta: float, optional, default=1.0
            A parameter to decide the radius for each points. The combination
            radius increases in proportion to this parameter.

        metric: str, optional, default='euclidean'
            The metric of each points. This parameter depends on the parameter
            `metric` of scipy.spatial.distance.pdist.

        t: 'inf' or float or int, optional, default='inf'
            The decay parameter of heat kernel. The weights are calculated as
            follow:

                W_{ij} = exp(-(||x_{i}-x_{j}||^2)/t)

            For more infomation, read the paper 'Laplacian Eigenmaps for
            Dimensionality Reduction and Data Representation', Belkin, et. al.

        include_self: bool, optional, default=True
            All diagonal elements are 1.0 if this parameter is True.

        is_sparse: bool, optional, default=True
            The method `cknneighbors_graph` returns csr_matrix object if this
            parameter is True else returns ndarray object.
    """

    def __init__(
        self,
        n_neighbors=5,
        delta=1.0,
        metric="euclidean",
        t="inf",
        include_self=False,
        is_sparse=True,
    ):
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.metric = metric
        self.t = t
        self.include_self = include_self
        self.is_sparse = is_sparse
        self.ckng = None

    def cknneighbors_graph(self, X):
        """A method to calculate the CkNN graph

        Args:
            X: ndarray
                The data matrix.

        return: csr_matrix (if self.is_sparse is True)
                or ndarray(if self.is_sparse is False)
        """

        n_neighbors = self.n_neighbors
        delta = self.delta
        metric = self.metric
        t = self.t
        include_self = self.include_self
        is_sparse = self.is_sparse

        n_samples = X.shape[0]

        if n_neighbors < 1 or n_neighbors > n_samples - 1:
            raise ValueError("`n_neighbors` must be in the range 1 to number of samples")
        if len(X.shape) != 2:
            raise ValueError("`X` must be 2D matrix")
        if n_samples < 2:
            raise ValueError("At least 2 data points are required")

        if metric == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("`X` must be square matrix")
            dmatrix = X
        else:
            dmatrix = squareform(pdist(X, metric=metric))

        darray_n_nbrs = np.partition(dmatrix, n_neighbors)[:, [n_neighbors]]
        ratio_matrix = dmatrix / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
        diag_ptr = np.arange(n_samples)

        if not isinstance(delta, (int, float)):
            raise ValueError("Invalid argument type. Type of `delta` must be float or int")
        adjacency = csr_matrix(ratio_matrix < delta)

        if include_self:
            adjacency[diag_ptr, diag_ptr] = True
        else:
            adjacency[diag_ptr, diag_ptr] = False

        if t == "inf":
            neigh = adjacency.astype(float)
        else:
            mask = adjacency.nonzero()
            weights = np.exp(-np.power(dmatrix[mask], 2) / t)
            dmatrix[:] = 0.0
            dmatrix[mask] = weights
            neigh = csr_matrix(dmatrix)

        if is_sparse:
            self.ckng = neigh
        else:
            self.ckng = neigh.toarray()

        return self.ckng
