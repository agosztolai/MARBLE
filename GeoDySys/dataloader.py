#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_cluster import random_walk

from torch_geometric.utils import dropout_adj
from torch_geometric.loader import NeighborSampler as NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, Batch

from GeoDySys.utils import fit_knn_graph


def loaders(data, n_neighbours, batch_size):
    
    train_loader = NeighborSampler(data.edge_index,
                             sizes=n_neighbours,
                             batch_size=batch_size,
                             shuffle=True,
                             num_nodes=data.num_nodes,
                             node_idx=data.train_mask)
    
    val_loader = NeighborSampler(data.edge_index,
                             sizes=n_neighbours,
                             batch_size=batch_size,
                             shuffle=False,
                             num_nodes=data.num_nodes,
                             node_idx=data.test_mask)
    
    test_loader = NeighborSampler(data.edge_index,
                             sizes=n_neighbours,
                             batch_size=batch_size,
                             shuffle=False,
                             num_nodes=data.num_nodes,
                             node_idx=data.test_mask)
    
    return train_loader, val_loader, test_loader


def construct_dataset(x, y, graph_type='knn', k=10):
        
    data_list = []
    for i, y_ in enumerate(y):
        #fit knn before adding function as node attribute
        x_ = torch.tensor(x[i], dtype=torch.float)
        edge_index = fit_knn_graph(x_, k=k)
        data_ = Data(x=x_, edge_index=edge_index)
        data_.pos = torch.tensor(x[i])
            
        #build pytorch geometric object
        data_.x = y_ #only function value as feature
        data_.num_nodes = len(x[i])
        data_.num_node_features = data_.x.shape[1]
        data_.y = torch.ones(data_.num_nodes, dtype=int)*i
        
        data_list.append(data_)
        
    #collate datasets
    batch = Batch.from_data_list(data_list)
    
    #split into training/validation/test datasets
    split = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    split(batch)
    
    return batch


class NeighborSampler(NeighborLoader):
    def __init__(self,*args,dropout=0.,**kwargs):
        super().__init__(*args,**kwargs)
        self.dropout=dropout
        
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()
        edge_index = torch.stack((row,col))
        edge_index, _ = dropout_adj(edge_index, p=self.dropout)
        row, col = edge_index

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch[:, 1], neg_batch], dim=0)
        
        return super(NeighborSampler, self).sample(batch)