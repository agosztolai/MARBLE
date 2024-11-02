"""Preprocessing module."""
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from procrustes import rotational

from MARBLE import geometry as g
from MARBLE import utils


def construct_dataset(
    anchor,
    vector,
    label=None,
    mask=None,
    graph_type="cknn",
    k=20,
    delta=1.0,
    n_eigenvalues=None,
    frac_geodesic_nb=1.5,
    spacing=0.0,
    number_of_resamples=1,
    var_explained=0.9,
    local_gauges=False,
    seed=None,
    pre_align=False,
):
    """Construct PyG dataset from node positions and features.

    Args:
        pos: matrix with position of points
        features: matrix with feature values for each point
        labels: any additional data labels used for plotting only
        mask: boolean array, that will be forced to be close (default is None)
        graph_type: type of nearest-neighbours graph: cknn (default), knn or radius
        k: number of nearest-neighbours to construct the graph
        delta: argument for cknn graph construction to decide the radius for each points.
        n_eigenvalues: number of eigenvalue/eigenvector pairs to compute (None means all, 
                       but this can be slow)
        frac_geodesic_nb: number of geodesic neighbours to fit the gauges to
        to map to tangent space k*frac_geodesic_nb
        stop_crit: stopping criterion for furthest point sampling
        number_of_resamples: number of furthest point sampling runs to prevent bias (experimental)
        var_explained: fraction of variance explained by the local gauges
        local_gauges: is True, it will try to compute local gauges if it can (signal dim is > 2,
            embedding dimension is > 2 or dim embedding is not dim of manifold)
        seed: Specify for reproducibility in the furthest point sampling. 
              The default is None, which means a random starting vertex.
    """
    
    # upper list is systems and sublist is the conditions
    anchor = [[torch.tensor(u).float() for u in sublist] for sublist in utils.to_list_of_lists(anchor)]
    vector = [[torch.tensor(u).float() for u in sublist] for sublist in utils.to_list_of_lists(vector)]
    num_node_features = vector[0][0].shape[1]

    if label is None:
        label = [[torch.arange(len(a)) for a in condition] for condition in utils.to_list_of_lists(anchor)]
    else:
        label = [torch.tensor(lab).float() for lab in utils.to_list_of_lists(label)]

    if mask is None:
        mask = [[torch.zeros(len(a), dtype=torch.bool) for a in condition] for condition in utils.to_list_of_lists(anchor)]
    else:
        mask = [torch.tensor(m) for m in utils.to_list_of_lists(mask)]

    if spacing == 0.0:
        number_of_resamples = 1

    data_list = []
    graph_id = 0
    for sys, (anchor_, vector_, label_, mask_) in enumerate(zip(anchor, vector, label, mask)):
        for con, (a,v,l,m) in enumerate(zip(anchor_, vector_, label_, mask_)):
            for _ in range(number_of_resamples):
                # even sampling of points
                if seed is None:
                    start_idx = torch.randint(low=0, high=len(a), size=(1,))
                else:
                    start_idx = 0
            
                sample_ind, _ = g.furthest_point_sampling(a, spacing=spacing, start_idx=start_idx)
                sample_ind, _ = torch.sort(sample_ind)  # this will make postprocessing easier
                a_, v_, l_, m_ = a[sample_ind], v[sample_ind], l[sample_ind], m[sample_ind]
                
                # fit graph to point cloud
                edge_index, edge_weight = g.fit_graph(a_, graph_type=graph_type, par=k, delta=delta)
    
                # define data object
                data_ = Data(
                    pos=a_,
                    x=v_,
                    pos_rotated=a_, # added for later
                    x_rotated=v_, # added for later
                    label=l_,
                    mask=m_,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    num_nodes=len(a_),
                    num_node_features=num_node_features,
                    y=torch.ones(len(a_), dtype=int) * con,                    
                    system=torch.ones(len(a_), dtype=int) * sys,    
                    condition=torch.ones(len(a_), dtype=int) * con, 
                    graph=torch.ones(len(a_), dtype=int) * graph_id, 
                    sample_ind=sample_ind,
                )
                
                data_list.append(data_)
                graph_id += 1


    # collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    batch.number_of_resamples = number_of_resamples
    batch.num_systems = len(anchor)
    batch.num_conditions = len(anchor[0])
    
    
    # for sys1, (anchor_) in enumerate(zip(anchor)):        
    #     l_gauges, Sigma = g.compute_gauges(batch, n_geodesic_nb=n_geodesic_nb)

    # compute procrustes for initialisation
    initial_rotation = []
    for sys1, (anchor_,) in enumerate(zip(anchor)):
        if pre_align:
            r = g.procrustes_analysis(torch.vstack(anchor_), torch.vstack(anchor[0])) # x rotated to y
        else:
            size = anchor[0][0].shape[1]
            #r = torch.eye(size)        
            r = torch.randn(size,size)   
        initial_rotation.append(r) # TODO check if r should be transpose

    
    batch.initial_rotations = initial_rotation

    # split into training/validation/test datasets
    split = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
    batch = split(batch)

    return _compute_geometric_objects(
        batch,
        local_gauges=local_gauges,
        n_geodesic_nb=k * frac_geodesic_nb,
        var_explained=var_explained,
    )


def _compute_geometric_objects(
    data,
    n_geodesic_nb=10,
    var_explained=0.9,
    local_gauges=False,
):
    """
    Compute geometric objects used later: local gauges, Levi-Civita connections
    gradient kernels, scalar and connection laplacians.

    Args:
        data: pytorch geometric data object
        n_geodesic_nb: number of geodesic neighbours to fit the tangent spaces to
        var_explained: fraction of variance explained by the local gauges
        local_gauges: whether to use local or global gauges

    Returns:
        data: pytorch geometric data object with the following new attributes
        kernels (list of d (nxn) matrices): directional kernels
        L (nxn matrix): scalar laplacian
        Lc (ndxnd matrix): connection laplacian
        gauges (nxdxd): local gauges at all points
        par (dict): updated dictionary of parameters
        local_gauges: whether to use local gauges

    """
    n, dim_emb = data.pos.shape
    dim_signal = data.x.shape[1]
    print(f"\n---- Embedding dimension: {dim_emb}", end="")
    print(f"\n---- Signal dimension: {dim_signal}", end="")

    # disable vector computations if 1) signal is scalar or 2) embedding dimension
    # is <= 2. In case 2), either M=R^2 (manifold is whole space) or case 1).
    if dim_signal == 1:
        print("\nSignal dimension is 1, so manifold computations are disabled!")
        local_gauges = False
    if dim_emb <= 2:
        print("\nEmbedding dimension <= 2, so manifold computations are disabled!")
        local_gauges = False
    if dim_emb != dim_signal:
        print("\nEmbedding dimension /= signal dimension, so manifold computations are disabled!")

    #if local_gauges:
    try:
        l_gauges, Sigma = g.compute_gauges(data, n_geodesic_nb=n_geodesic_nb)
    except Exception as exc:
        raise Exception(
            "\nCould not compute gauges (possibly data is too sparse or the \
              number of neighbours is too small)"
        ) from exc
    #else:
    global_gauges = torch.eye(dim_emb).repeat(n, 1, 1)

    L = g.compute_laplacian(data)

    dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
    print(f"---- Manifold dimension: {dim_man}")

    l_gauges = l_gauges[:, :, :dim_man]
    R = g.compute_connections(data, l_gauges)
    Lc = g.compute_connection_laplacian(data, R)


    if dim_man < dim_signal:
        l_gauges, normal_vectors = g.flip_gauges(data, l_gauges)
    else:
        normal_vectors = []

    if local_gauges:
        
        data.dim_man = dim_man        

        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, l_gauges)
        kernels = [utils.tile_tensor(K, data.dim_man) for K in kernels]
        kernels = [K * R for K in kernels]


    else:
        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, global_gauges)
        #Lc = None

    # print("\n---- Computing eigendecomposition ... ", end="")
    L = g.compute_eigendecomposition(L)
    Lc = g.compute_eigendecomposition(Lc)

    data.kernels = [
        utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values()) for K in kernels
    ]
    data.L, data.Lc, data.gauges, data.l_gauges, data.normal_vectors = (L,
                                                                             Lc,
                                                                             global_gauges,
                                                                             l_gauges,
                                                                             normal_vectors)

    return data
