"""Preprocessing module."""
import torch

from .lib import geometry as g
from .lib import utils


def preprocessing(
    data,
    n_geodesic_nb=2.0,
    var_explained=0.9,
    n_evec=2,
    diffusion_method="spectral",
    vector=True,
    compute_laplacian=False,
    compute_connection_laplacian=False,
    dim_man=None,
):
    """
    Compute geometric objects used later: local gauges, Levi-Civita connections
    gradient kernels, scalar and connection laplacians.

    Parameters
    ----------
    data : pytorch geometric data object
    n_geodesic_nb: number of geodesic neighbours to fit the gauges to
    to map to tangent space
    var_explained: fraction of variance explained by the local gauges
    diffusion: 'spectral' or 'matrix_exp'

    Returns
    -------
    R : (nxnxdxd) tensor of L-C connectionc (dxd) matrices
    kernels : list of d (nxn) matrices of directional kernels
    L : (nxn) matrix of scalar laplacian
    Lc : (ndxnd) matrix of connection laplacian
    par : updated dictionary of parameters

    """

    n, dim_emb = data.pos.shape
    dim_signal = data.x.shape[1]
    print(f"---- Embedding dimension: {dim_emb}")
    print(f"---- Signal dimension: {dim_signal}\n")

    # disable vector computations if 1) signal is scalar or 2) embedding dimension
    # is <= 2. In case 2), either M=R^2 (manifold is whole space) or case 1).
    if not vector:
        local_gauges = False
        print("Vector computations are disabled")
    elif dim_signal == 1:
        print("Signal dimension is 1, so manifold computations are disabled!")
        local_gauges = False
    elif dim_emb <= 2:
        print("Embedding dimension <= 2, so manifold computations are disabled!")
        local_gauges = False
    elif dim_emb != dim_signal:
        print("Embedding dimension /= signal dimension, so manifold computations are disabled!")
    else:
        local_gauges = True

    # gauges
    if local_gauges:
        try:
            gauges, Sigma = g.compute_gauges(data, n_geodesic_nb=n_geodesic_nb)
        except Exception as exc:
            raise Exception(
                "\nCould not compute gauges (possibly data is too sparse or the \
                  number of neighbours is too small)"
            ) from exc
    else:
        gauges = torch.eye(dim_emb).repeat(n, 1, 1)

    # Laplacian
    if compute_laplacian:
        L = g.compute_laplacian(data)
    else:
        L = None

    if local_gauges:
        if not dim_man:
            dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
        data.dim_man = dim_man

        print(f"\n---- Manifold dimension: {dim_man}")

        gauges = gauges[:, :, :dim_man]
        R = g.compute_connections(data, gauges)

        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        kernels = [utils.tile_tensor(K, dim_man) for K in kernels]
        kernels = [K * R for K in kernels]
        print("Done ")

        if compute_connection_laplacian:
            Lc = g.compute_connection_laplacian(data, R)
        else:
            Lc = None

    else:
        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        print("Done ")
        Lc = None

    if diffusion_method == "spectral":
        print("---- Computing eigendecomposition ... ", end="")
        L = g.compute_eigendecomposition(L)
        Lc = g.compute_eigendecomposition(Lc)
        print("Done ")

    data.kernels = [
        utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values()) for K in kernels
    ]
    data.L, data.Lc, data.gauges, data.local_gauges = L, Lc, gauges, local_gauges

    return data
