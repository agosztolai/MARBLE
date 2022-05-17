#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor
from .utils import adjacency_matrix

"""Convolution"""
class AnisoConv(MessagePassing):    
    def __init__(self, 
                 adj_norm=False, 
                 **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.adj_norm = adj_norm

    def forward(self, x, edge_index, K=None):
        """Forward pass"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
            
        size = (len(x[0]), len(x[1]))
        if K is not None: #anisotropic kernel
            out = []
            #evaluate all directional kernels and concatenate results columnwise
            for K_ in K:
                K_ = adjacency_matrix(edge_index, size, value=K_.t())
                out.append(self.propagate(K_.t(), x=x))
            out = torch.cat(out, axis=1)
            
        else: #use adjacency matrix (vanilla GCN)
            out = self.propagate(edge_index, x=x)
        
        if self.adj_norm: #normalize features by the mean of neighbours
            adj = adjacency_matrix(edge_index, size)
            out = adj_norm(x[0], out, adj.t())

        out += x[1].repeat([1,out.shape[1]//x[1].shape[1]]) #add back root nodes
            
        return out

    def message_and_aggregate(self, K_t, x):
        """Anisotropic convolution step. Need to be transposed because of PyG 
        convention. This is executed if input to propagate() is a SparseTensor"""
        return matmul(K_t, x[0], reduce=self.aggr)
    
    def message(self, x_j, edge_weight):
        """Convolution step. This is executed if input to propagate() is 
        an edge list tensor"""
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
def adj_norm(x, out, adj_t):
    """Normalize features by mean of neighbours"""
    ones = torch.ones([x.shape[0],1])
    # x = x.norm(dim=-1,p=2, keepdim=True)
    mu_x = matmul(adj_t, x) / matmul(adj_t, ones)
    # sigma_x = (matmul(adj_t, x**2) / matmul(adj_t, ones)) - mu_x**2
    out -= mu_x#.repeat([1,out.shape[1]//x.shape[1]])
    # out /= (sigma_x)**(.5)
    # out[torch.isnan(out)]=0
    
    return out