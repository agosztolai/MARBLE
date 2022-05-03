#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import matmul, SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor
import torch.nn.functional as F

"""Main network architecture"""
class net(nn.Module):
    def __init__(self, data, hidden_channels, out_channels=None, **kwargs):
        super(net, self).__init__()
        
        self.n_layers = kwargs['n_layers'] if 'n_layers' in kwargs else 1
        
        #initialize conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(self.n_layers):
            self.convs.append(AnisoConv(
                adj_norm=kwargs['adj_norm'] if 'adj_norm' in kwargs else False))
        
        #initialize multilayer perceptron
        self.MLP = MLP(in_channels=data.x.shape[1] if not hasattr(data, 'kernels') else len(data.kernels)*data.x.shape[1],
                       hidden_channels=hidden_channels, 
                       out_channels=out_channels,
                       n_lin_layers=kwargs['n_lin_layers'] if 'n_lin_layers' in kwargs else 1,
                       activation=kwargs['activation'] if 'activation' in kwargs else False,
                       dropout=kwargs['dropout'] if 'dropout' in kwargs else 0.,
                       b_norm=kwargs['b_norm'] if 'b_norm' in kwargs else False)
        
    def forward(self, x, adjs, K=None):
        """forward pass"""
        #loop over minibatches
        for i, (edge_index, _, size) in enumerate(adjs):
            #messages are passed from x_source (all nodes) to x_target
            x_source = x 
            x_target = x[:size[1]]
            
            #convolution
            x = self.convs[i]((x_source, x_target), edge_index, K=K, edge_weight=None, size=size)
            
            #multilayer perceptron
            x = self.MLP(x)
                                              
        return x

    # for testing, we don't do minibatch
    def full_forward(self, x, edge_index, K=None):        
        for conv in self.convs:
            x = conv(x, edge_index, K=K, size=(x.shape[0],x.shape[0]))
            x = self.MLP(x)
            
        return x
    
"""Convolution"""
class AnisoConv(MessagePassing):    
    def __init__(self, adj_norm=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.adj_norm = adj_norm

    def forward(self, x, edge_index, K=None, edge_weight=None, size=None):
        """forward pass"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
            
        if K is not None: #anisotropic kernel
            out = []
            #evaluate all directional kernels and concatenate results columnwise
            for K_ in K:
                K_ = K_.t()
                K_ = SparseTensor(row=edge_index[0], col=edge_index[1], 
                                  value=K_[edge_index[0],edge_index[1]],
                                  sparse_sizes=(size[0], size[1]))
                out.append(self.propagate(K_.t(), x=x, edge_weight=edge_weight, size=size))
            
            out = torch.cat(out, axis=1)

        else: #use adjacency matrix (vanilla GCN)
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        
        if self.adj_norm: #normalize features by the mean of neighbours
            adj = self.adjacency_matrix(edge_index, size)
            out = self.adj_norm_(x[0], out, adj.t())


        out += torch.tile(x[1], (1, 2)) #add back root nodes
            
        return out
    
    def adj_norm_(self, x, out, adj_t, norm=1):
        """Normalize features by mean of neighbours"""

        ones = torch.ones_like(x)
        x = x**norm
        norm_x = matmul(adj_t, x) / matmul(adj_t, ones)
        norm_x = torch.tile(norm_x, (1, 2))
        out -= norm_x
        
        return out
    
    def adjacency_matrix(self, edge_index, size):
        """Compute adjacency matrix from edge_index"""
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=None,
                sparse_sizes=(size[0], size[1]))
        
        return adj

    def message_and_aggregate(self, K_t, x):
        """Anisotropic convolution step. Need to be transposed because of PyG 
        convention. This is executed if input to propagate() is a SparseTensor"""
        return matmul(K_t, x[0], reduce=self.aggr)
    
    def message(self, x_j, edge_weight):
        """Convolution step. This is executed if input to propagate() is 
        an edge list tensor"""
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    
"""Multi-layer perceptron composed of linear layers, activation and batch norm"""
class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 n_lin_layers=1,
                 activation=True,
                 b_norm=False,
                 dropout=0.):
        super(MLP, self).__init__()
        
        if n_lin_layers <= 1:
            out_channels=hidden_channels
            hidden_channels=in_channels
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels=hidden_channels
            
        #stack linear layers
        self.lin = nn.ModuleList()
        for _ in range(n_lin_layers - 1):
            self.lin.append(Linear(in_channels, hidden_channels, bias=True))
  
        self.lin.append(Linear(hidden_channels, out_channels, bias=False))       
        self.activation = nn.ReLU() if activation else None
        self.dropout = nn.Dropout(dropout)
        self.b_norm = (lambda out: F.normalize(out, p=2., dim=-1)) if b_norm else False
        
        self.init_fn = nn.init.xavier_uniform_
        self.reset_parameters(in_channels)
        self.n_lin_layers = n_lin_layers
        
    def reset_parameters(self, in_channels):
        """Initialise parameters"""
        for l in self.lin:
            l.reset_parameters()
            if self.init_fn is not None:
                self.init_fn(l.weight, 1/in_channels)
            if l.bias is not None:
                l.bias.data.zero_()
        
    def forward(self, x):
        """Forward pass"""
        for i in range(self.n_lin_layers):
            x = self.lin[i](x)
            if (self.activation is not None) and (i+1!=self.n_lin_layers):
                x = self.activation(x)
            x = self.dropout(x)
            if self.b_norm:
                x = self.b_norm(x)
        return x