import logging

import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh

from ptu_dijkstra import ptu_dijkstra


class PTU():
    """
    Parallel Transport Unfolding class

    Parameters
    ----------
    X: numpy matrix
        (N, D) matrix of N input data points in D dimensional space sampling a
        lower dimensional manifold S
    n_neighbors : int, optional
        Number of nearest neighbors to use for graph construction. Eucledean
        distances between nearest neighbors are assumed to be good
        approximations of true geodesic distances between them.
    geod_n_neighbors : int
        Number of points to include in geodesic neighborhoods. Geodesic
        neighborhood of a point x of size K is K nearest neighbors to x in
        the proximity graph. Notice it's different than simple K nearest
        neighbors in ambient D dimensional space. Geodesic neighborhood of
        point x is used to compute local tangent space to the data manifold at
        x. Note that geod_n_neighbors must be greater or equal to n_neighbors.
    embedding_dim : int
        Embedding dimensionality, i.e. true or estimated dimension of
        manifold S. Note that embedding_dim mush be less or equal to the
        ambient dimension D of S (where D = X.shape[1])
    verbose : bool, optional
        If True, prints out logging info.
    n_jobs: int, optional
        Number of jobs to run in parallel to compute sparse kNN graph.
    """
    def __init__(
            self,
            X,
            n_neighbors,
            geod_n_neighbors,
            embedding_dim,
            verbose=False,
            n_jobs=None):
        self.X = X
        self.n_neighbors = n_neighbors
        self.geod_n_neighbors = geod_n_neighbors
        self.embedding_dim = embedding_dim
        self.graph = None
        self.ptu_dists = None
        self.n_jobs = n_jobs
        self.Embedding = None
        self.initialize_logger(verbose)
        self.validate_dimensions()

    @classmethod
    def with_custom_graph(
            cls,
            X,
            graph,
            geod_n_neighbors,
            embedding_dim,
            verbose=False):
        """
        Constructor that uses custom proximity graph instead of computing kNN
        graph.
        """
        assert (
                (graph.shape[0] == graph.shape[1]) and
                (graph.shape[0] == X.shape[0])
            ), \
            'Graph dimensions do not agree with pointset dimensions'

        instance = cls(
            X=X,
            n_neighbors=None,
            geod_n_neighbors=geod_n_neighbors,
            embedding_dim=embedding_dim,
            verbose=verbose,
            n_jobs=None
        )
        instance.graph = graph
        return instance

    def initialize_logger(self, verbose):
        """
        Setting up logger format
        """
        logger = logging.getLogger(__name__)
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)
        self.logger = logger

    def validate_dimensions(self):
        """
        Validates input parameters related to dimensions
        """
        if self.X.shape[1] < self.embedding_dim:
            raise ValueError(
                "Embedding dimension must be less or equal to the ambient "
                "dimension of input data"
            )
        if self.geod_n_neighbors >= self.X.shape[0]:
            raise ValueError(
                "Geodesic neighborhood size must be less than the "
                "total number of samples"
            )
        if ((self.n_neighbors is not None) and
                (self.n_neighbors >= self.X.shape[0])):
            raise ValueError(
                "kNN neighborhood size must be less than the "
                "total number of samples"
            )
        if ((self.n_neighbors is not None) and
                (self.geod_n_neighbors < self.n_neighbors)):
            raise ValueError(
                "Geodesic neighborhood size should be larger or equal to the "
                "n_neighbors"
            )
        if self.geod_n_neighbors < self.embedding_dim:
            raise ValueError(
                "Geodesic neighborhood size must be larger or equal to the "
                "embedding dimension"
            )

    def fit(self):
        """
        Parallel Transport Unfolding dimensionality reduction procedure
        """
        self.compute_geodesic_distances()
        self.mds()
        return self.Embedding

    def compute_proximity_graph(self):
        if self.graph is None:
            self.logger.info('Constructing proximity graph on {} points in {}D'
                             .format(self.X.shape[0], self.X.shape[1]))
            nn = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                n_jobs=self.n_jobs
            )
            nn.fit(self.X)
            self.graph = nn.kneighbors_graph(mode='distance')
            self.logger.info('Constructing proximity graph: done')
        else:
            self.logger.info('Using provided proximity graph')

    def compute_geodesic_distances(self):
        """
        Parallel Transport Unfolding pairwise geodesic distance calculation
        """
        try:
            self.compute_proximity_graph()
            self.logger.info('Computing pairwise parallel transport distances')
            ptu_dists = ptu_dijkstra(
                X=self.X,
                csgraph=self.graph,
                d=self.embedding_dim,
                K=self.geod_n_neighbors
            )
            self.ptu_dists = 0.5 * (ptu_dists + ptu_dists.T)
            self.logger.info(
                'Computing pairwise parallel transport distances: done'
            )
        except Exception:
            self.logger.exception('Geodesic distance computation failed')
            raise

    def mds(self):
        """
        MultiDimensional Scaling
        """
        try:
            N = self.ptu_dists.shape[0]
            self.logger.info(
                'Performing MultiDimensional Scaling on {0}x{0} matrix'
                .format(N)
            )
            G = -0.5 * np.square(self.ptu_dists)
            col_mean = np.sum(G, axis=0) / N
            row_mean = (np.sum(G, axis=1) / N)[:, np.newaxis]
            mean = col_mean.sum() / N
            G = G - col_mean - row_mean + mean

            if N > 200 and self.embedding_dim < 10:
                # sparse arpack eigensolver
                S, V = eigsh(G, self.embedding_dim, which="LA")
            else:
                # dense eigensolver
                S, V = eigh(G, eigvals=[N-self.embedding_dim, N-1])
            self.Embedding = V * np.sqrt(np.abs(S))
            self.logger.info('Performing MultiDimensional Scaling: done')
        except Exception:
            self.logger.exception('MDS computation failed')
            raise
