"""Data loader module."""
import torch
from torch_cluster import random_walk
from torch_geometric.loader import NeighborSampler as NS


def loaders(data, par):
    """Loaders."""
    nb = [par["n_sampled_nb"]] * par["order"]

    train_loader = NeighborSampler(
        edge_index=data.edge_index,
        sizes=nb,
        batch_size=par["batch_size"],
        shuffle=True,
        num_nodes=data.num_nodes,
        node_idx=data.train_mask,
        system_index=data.system,
        condition_index=data.condition,
    )

    val_loader = NeighborSampler(
        edge_index=data.edge_index,
        sizes=nb,
        batch_size=par["batch_size"],
        shuffle=True,
        num_nodes=data.num_nodes,
        node_idx=data.val_mask,
        system_index=data.system,
        condition_index=data.condition,
    )

    test_loader = NeighborSampler(
        edge_index=data.edge_index,
        sizes=nb,
        batch_size=par["batch_size"],
        shuffle=True,
        num_nodes=data.num_nodes,
        node_idx=data.test_mask,
        system_index=data.system,
        condition_index=data.condition,
    )

    return train_loader, val_loader, test_loader


class NeighborSampler(NS):
    """Neighbor Sampler."""
    def __init__(self, system_index=None, condition_index=None, **kwargs):
        super().__init__(**kwargs)  # Pass all the unnamed and named arguments to the parent class
        self.system_index = system_index  # Additional initialization for the child class
        self.condition_index = condition_index  # Additional initialization for the child class
     
    def sample(self, batch):
        """Sample."""
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # sample) and a random node (as negative sample):
        batch = torch.tensor(batch)
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)
        neg_batch = sample_neg_nodes(batch, self.system_index, col, row)        
        
        #neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(),))
        batch = torch.cat([batch, pos_batch[:, 1], neg_batch], dim=0)

        return super().sample(batch)


def sample_neg_nodes(batch_nodes, system_index, row, col):
    """ only sample negative nodes from within the same graph """
    # Initialize tensor to hold negative samples for the batch
    neg_samples = torch.empty_like(batch_nodes)
    
    # Iterate over each node in the batch
    for i, node in enumerate(batch_nodes):
        # Nodes in the same system as the current node
        same_graph_nodes = (system_index == system_index[node]).nonzero(as_tuple=False).view(-1)
        
        # Find neighbours of node
        neighbors = col[row==node]
        
        # Exclude the current node to avoid it being selected as a negative sample
        nodes_to_exclude = torch.cat((neighbors, node.unsqueeze(0)), dim=-1)
        
        choices = same_graph_nodes[~torch.isin(same_graph_nodes,nodes_to_exclude)]

        # Randomly select a negative sample from the available choices
        if choices.numel() > 0:
            neg_samples[i] = choices[torch.randint(0, choices.size(0), (1,))]
        else:
            # Fallback if no other nodes are in the same graph (unlikely in practice for large graphs)
            neg_samples[i] = node
    
    return neg_samples

# =============================================================================
# below is an alternative implementation, not working yet
# =============================================================================

# from torch_geometric.loader import LinkNeighborLoader
# from torch_geometric.utils import subgraph

# def loaders(data, par):

#     nb = [par['n_sampled_nb'] for i in range(max(par['order'], par['depth']))]

#     train_loader = LinkNeighborLoader(
#         data,
#         num_neighbors=nb,
#         shuffle=True,
#         batch_size=par['batch_size'],
#         edge_label_index=subgraph(data.train_mask, data.edge_index)[0],
#         neg_sampling_ratio=1
#     )

#     val_loader = LinkNeighborLoader(
#         data,
#         num_neighbors=nb,
#         shuffle=False,
#         batch_size=par['batch_size'],
#         edge_label_index=subgraph(data.val_mask, data.edge_index)[0],
#         neg_sampling_ratio=1
#     )

#     test_loader = LinkNeighborLoader(
#         data,
#         num_neighbors=nb,
#         shuffle=False,
#         batch_size=par['batch_size'],
#         edge_label_index=subgraph(data.test_mask, data.edge_index)[0],
#         neg_sampling_ratio=1
#     )

#     return train_loader, val_loader, test_loader
