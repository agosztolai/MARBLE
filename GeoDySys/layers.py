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
                 root_weight=True,
                 eps=0.,
                 **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.adj_norm = adj_norm
        self.root_weight = root_weight
        
        #uf root_weight=True then eps is a trainable parameter
        if root_weight:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        self.reset_parameters(eps)

    def reset_parameters(self, eps):
        self.eps.data.fill_(eps)

    def forward(self, x, edge_index, K=None):
        """Forward pass"""
        if isinstance(x, Tensor):
            #when there is no minibatching, messages are passed from all source
            #to all target nodes
            x: OptPairTensor = (x, x)
            
        if not isinstance(K, list):
            K = [K]
            
        size = (len(x[0]), len(x[1]))
        out = []
        #evaluate all directional kernels and concatenate results columnwise
        for K_ in K:
            if K_ is not None: #anisotropic kernel
                K_ = adjacency_matrix(edge_index, size, value=K_.t())
            else: #adjacency matrix (vanilla GCN)
                K_ = adjacency_matrix(edge_index, size, value=None)
                
            out_ = self.propagate(K_.t(), x=x[0])
                
            #skip connection
            out_ += self.eps * x[1] 
                
            if self.adj_norm: #adjacency features
                adj = adjacency_matrix(edge_index, size)
                out_ = adj_norm(x[0], out_, adj.t(), K_.t(), float(self.eps))
                    
            out.append(out_)
                    
        out = torch.cat(out, axis=1)
  
        return out

    def message_and_aggregate(self, K_t, x):
        """Anisotropic convolution step. Need to be transposed because of PyG 
        convention. This is executed if input to propagate() is a SparseTensor"""
        return matmul(K_t, x, reduce=self.aggr)
    
    
def adj_norm(x, out, adj_t, K_t, eps):
    """Normalize features by mean of neighbours"""
    ones = torch.ones([x.shape[0],1])
    # x = x.norm(dim=-1,p=2, keepdim=True)
    mu_x = (matmul(adj_t, x) + eps*out) / (matmul(adj_t, ones) + (eps>0)*1)
    K1 = matmul(K_t, ones)
    # sigma_x = (matmul(adj_t, x**2) / matmul(adj_t, ones)) - mu_x**2
    out -= (K1*mu_x)#.repeat([1,out.shape[1]//x.shape[1]])
    
    return out