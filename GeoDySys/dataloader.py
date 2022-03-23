#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_geometric.utils import dropout_adj
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk


class NeighborSampler(RawNeighborSampler):
    def __init__(self,*args,dropout=0.1,**kwargs):
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
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        
        return super(NeighborSampler, self).sample(batch)