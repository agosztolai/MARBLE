"""Preprocessing module."""
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from MARBLE import geometry as g
from MARBLE import utils


def construct_dataset(
    pos,
    features,
    labels=None,
    mask=None,
    graph_type="cknn",
    k=20,
    frac_geodesic_nb=1.5,
    stop_crit=0.0,
    number_of_resamples=1,
    var_explained=0.9,
    local_gauges=False,
    delta=1.0,
):
    """Construct PyG dataset from node positions and features.

    Args:
        pos: matrix with position of points
        features: matrix with feature values for each point
        labels: any additional data labels used for plotting only
        graph_type: type of nearest-neighbours graph: cknn (default), knn or radius
        k: number of nearest-neighbours to construct the graph
        frac_geodesic_nb: number of geodesic neighbours to fit the gauges to
        to map to tangent space k*frac_geodesic_nb
        stop_crit: stopping criterion for furthest point sampling
        number_of_resamples: number of furthest point sampling runs to prevent bias (experimental)
        var_explained: fraction of variance explained by the local gauges
        local_gauges: is True, it will try to compute local gauges if it can (signal dim is > 2,
            embedding dimension is > 2 or dim embedding is not dim of manifold)
        delta: argument for cknn graph construction to decide the radius for each points.
    """

    pos = [torch.tensor(p).float() for p in utils.to_list(pos)]
    features = [torch.tensor(x).float() for x in utils.to_list(features)]
    num_node_features = features[0].shape[1]
    
    if labels is None:
        labels = [torch.arange(len(p)) for p in utils.to_list(pos)]
    else:
        labels = [torch.tensor(l).float() for l in utils.to_list(labels)]
        
    if mask is None:
        mask = [torch.zeros(len(p), dtype=torch.bool) for p in utils.to_list(pos)]
    else:
        mask = [torch.tensor(m) for m in utils.to_list(mask)]

    if stop_crit == 0.0:
        number_of_resamples = 1

    data_list = []
    for i, (p, f, l, m) in enumerate(zip(pos, features, labels, mask)):
        for _ in range(number_of_resamples):
            # even sampling of points
            start_idx = torch.randint(low=0, high=len(p), size=(1,))
            sample_ind, _ = g.furthest_point_sampling(p, stop_crit=stop_crit, start_idx=start_idx)
            sample_ind, _ = torch.sort(sample_ind) #this will make postprocessing easier
            p_, f_, l_, m_ = p[sample_ind], f[sample_ind], l[sample_ind], m[sample_ind]

            # fit graph to point cloud
            edge_index, edge_weight = g.fit_graph(p_, graph_type=graph_type, par=k, delta=delta)
            n = len(p_)
            data_ = Data(
                pos=p_,
                x=f_,
                l=l_,
                mask=m_,
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=n,
                num_node_features=num_node_features,
                y=torch.ones(n, dtype=int) * i,
                sample_ind=sample_ind,
            )

            data_list.append(data_)

    # collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    batch.number_of_resamples = number_of_resamples

    # split into training/validation/test datasets
    split = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
    split(batch)

    return _compute_geometric_objects(
        batch,
        local_gauges=local_gauges,
        frac_geodesic_nb=frac_geodesic_nb,
        var_explained=var_explained,
    )


def _compute_geometric_objects(data,
    frac_geodesic_nb=2.0,
    var_explained=0.9,
    local_gauges=False,
):
    """
    Compute geometric objects used later: local gauges, Levi-Civita connections
    gradient kernels, scalar and connection laplacians.

    Args:
        data: pytorch geometric data object
        n_geodesic_nb: number of geodesic neighbours to fit the gauges to map to tangent space
        var_explained: fraction of variance explained by the local gauges
        diffusion: 'spectral' or 'matrix_exp'

    Returns:
        R (nxnxdxd tensor): L-C connectionc (dxd) matrices
        kernels (list of d (nxn) matrices): directional kernels
        L (nxn matrix): scalar laplacian
        Lc (ndxnd matrix): connection laplacian
        par (dict): updated dictionary of parameters
    """
    n, dim_emb = data.pos.shape
    dim_signal = data.x.shape[1]
    print(f"---- Embedding dimension: {dim_emb}")
    print(f"---- Signal dimension: {dim_signal}\n")

    # disable vector computations if 1) signal is scalar or 2) embedding dimension
    # is <= 2. In case 2), either M=R^2 (manifold is whole space) or case 1).
    if dim_signal == 1:
        print("Signal dimension is 1, so manifold computations are disabled!")
        local_gauges = False
    if dim_emb <= 2:
        print("Embedding dimension <= 2, so manifold computations are disabled!")
        local_gauges = False
    if dim_emb != dim_signal:
        print("Embedding dimension /= signal dimension, so manifold computations are disabled!")

    if local_gauges:
        try:
            gauges, Sigma = g.compute_gauges(data, n_geodesic_nb=frac_geodesic_nb)
        except Exception as exc:
            raise Exception(
                "\nCould not compute gauges (possibly data is too sparse or the \
                  number of neighbours is too small)"
            ) from exc
    else:
        gauges = torch.eye(dim_emb).repeat(n, 1, 1)

    L = g.compute_laplacian(data)

    if local_gauges:
        data.dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
        print(f"\n---- Manifold dimension: {data.dim_man}")

        gauges = gauges[:, :, :data.dim_man]
        R = g.compute_connections(data, gauges)

        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        kernels = [utils.tile_tensor(K, data.dim_man) for K in kernels]
        kernels = [K * R for K in kernels]

        Lc = g.compute_connection_laplacian(data, R)

    else:
        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        Lc = None

    print("---- Computing eigendecomposition ... ", end="")
    L = g.compute_eigendecomposition(L)
    Lc = g.compute_eigendecomposition(Lc)

    data.kernels = [
        utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values()) for K in kernels
    ]
    data.L, data.Lc, data.gauges, data.local_gauges = L, Lc, gauges, local_gauges

    return data
