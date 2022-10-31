#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch_cluster import random_walk

from torch_geometric.loader import NeighborSampler as NS


def loaders(data, par, split=True):
    
    nb = [par['n_sampled_nb']] * par['order']
    
    if split:
        train_loader = NeighborSampler(data.edge_index,
                                       sizes=nb,
                                       batch_size=par['batch_size'],
                                       shuffle=True,
                                       num_nodes=data.num_nodes,
                                       node_idx=data.train_mask)
        
        val_loader = NeighborSampler(data.edge_index,
                                     sizes=nb,
                                     batch_size=par['batch_size'],
                                     shuffle=False,
                                     num_nodes=data.num_nodes,
                                     node_idx=data.val_mask)
        
        test_loader = NeighborSampler(data.edge_index,
                                      sizes=nb,
                                      batch_size=par['batch_size'],
                                      shuffle=False,
                                      num_nodes=data.num_nodes,
                                      node_idx=data.test_mask)
        
        return train_loader, val_loader, test_loader
    else:
        loader = NeighborSampler(data.edge_index,
                                 sizes=nb,
                                 batch_size=par['batch_size'],
                                 shuffle=True)
        
        return loader


class NeighborSampler(NS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
    def sample(self, batch):
        
        row, col, _ = self.adj_t.coo()
        
        # For each node in `batch`, we sample a direct neighbor (as positive
        # sample) and a random node (as negative sample):
        batch = torch.tensor(batch)
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)
        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ))
        batch = torch.cat([batch, pos_batch[:, 1], neg_batch], dim=0)
        
        return super(NeighborSampler, self).sample(batch)
    
    
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