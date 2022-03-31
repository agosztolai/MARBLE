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


class net(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, **kwargs):
        super(net, self).__init__()#count add (aggr='mean') to define aggregation function
        self.num_layers = num_layers
        self.adj_norm = kwargs['adj_norm'] if 'adj_norm' in kwargs else False
        
        #initialize conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(num_layers):
            in_channels=in_channels if i==0 else hidden_channels
            self.convs.append(
                AnisoConv(in_channels, 
                          hidden_channels, 
                          activation=kwargs['activation'] if 'activation' in kwargs else False,
                          dropout=kwargs['dropout'] if 'dropout' in kwargs else 0.,
                          b_norm=kwargs['b_norm'] if 'b_norm' in kwargs else False))

        #introduce parameter to learn linear combination of kernels
        # self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        
    #forward computation, input is the signal and the graph
    def forward(self, x, adjs, K=None):
        
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes first
            
            if K is not None:
                kernel = SparseTensor(row=edge_index[0], col=edge_index[1], 
                                  value=K[edge_index[0],edge_index[1]],
                                  sparse_sizes=(size[0], size[1]))    
            else:
                kernel = self.adjacency_matrix(edge_index, size)
                
            #perform convolution
            out = self.convs[i]((x, x_target), kernel.t(), edge_weight=None)
            
            #normalize each aggregated value by the mean of neighbours
            if self.adj_norm:
                adj = self.adjacency_matrix(edge_index, size)
                x = self.adj_norm_(x, out, adj.t())
            else:
                x = out
                                                
        return x
    
    def adj_norm_(self, x, out, adj_t, norm=1):
        ones = torch.ones_like(x)
        x = x**norm
        norm_x = matmul(adj_t, x) / matmul(adj_t, ones)
        
        out -= norm_x
        
        return out
    
    def adjacency_matrix(self, edge_index, size):
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=None,
                sparse_sizes=(size[0], size[1]))
        
        return adj

    # for testing, we don't do minibatch
    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            adj = self.adjacency_matrix(edge_index, (x.shape[0],x.shape[0]))
            
            if self.adj_norm:
                x = self.adj_norm_(x, x, adj.t())
            
        return x
    
    
class AnisoConv(MessagePassing):    
    r"""

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        activation=False,
        b_norm=False,
        dropout=0.,
        **kwargs,
    ):
        super().__init__(aggr='add', **kwargs)
        
        self.lin = Linear(in_channels, out_channels, bias=True)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        
        self.activation = nn.ReLU() if activation else False
        self.dropout = nn.Dropout(dropout)
        self.b_norm = (lambda out: F.normalize(out, p=2., dim=-1)) if b_norm else False

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, K_t, edge_weight=None, size=None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(K_t, x=x, edge_weight=edge_weight, size=size)
        out = self.lin(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)
            
        if self.activation:
            out = self.activation(out)
        out = self.dropout(out)
        if self.b_norm:
            out = self.b_norm(out)

        return out

    def message_and_aggregate(self, K_t, x):
        return matmul(K_t, x[0], reduce=self.aggr)
    
    # def message(self, x_j, edge_weight):
    #     return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    
# class MLP(nn.Module):
#     """
#     Multi-layer perceptron composed of linear layers
#     """

#     def __init__(self, in_size, hidden_size, out_size, layers,
#                  dropout=0., b_norm=False):
#         super(MLP, self).__init__()

#         self.in_size = in_size
#         self.hidden_size = hidden_size
#         self.out_size = out_size

#         self.MLP = nn.ModuleList()
#         if layers <= 1:
#             self.MLP.append(linear_layer(in_size, out_size, activation=False, b_norm=b_norm,
#                                                 dropout=dropout))
#         else:
#             self.MLP.append(linear_layer(in_size, hidden_size, activation=True, b_norm=b_norm,
#                                                 dropout=dropout))
#             for _ in range(layers - 2):
#                 self.MLP.append(linear_layer(hidden_size, hidden_size, activation=True,
#                                                     b_norm=b_norm, dropout=dropout))
#             self.MLP.append(linear_layer(hidden_size, out_size, activation=None, b_norm=b_norm,
#                                                 dropout=dropout))

#     def forward(self, x):
#         for fc in self.MLP:
#             x = fc(x)
#         return x

#     # def __repr__(self):
#     #     return self.__class__.__name__ + ' (' \
#     #            + str(self.in_size) + ' -> ' \
#     #            + str(self.out_size) + ')'
               

# class linear_layer(nn.Module):
#     """
#     Linear layer. This layer is centered around a torch.nn.Linear module.

#     Arguments
#     ----------
#         in_size: int
#             Input dimension of the layer (the torch.nn.Linear)
#         out_size: int
#             Output dimension of the layer.
#         dropout: float, optional
#             The ratio of units to dropout. No dropout by default.
#             (Default value = 0.)
#         b_norm: bool, optional
#             Whether to use batch normalization
#             (Default value = False)
#         bias: bool, optional
#             Whether to enable bias in for the linear layer.
#             (Default value = True)
#         init_fn: callable, optional
#             Initialization function to use for the weight of the layer. Default is
#             :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
#             (Default value = None)

#     Attributes
#     ----------
#         dropout: int
#             The ratio of units to dropout.
#         b_norm: int
#             Whether to use batch normalization
#         linear: torch.nn.Linear
#             The linear layer
#         init_fn: function
#             Initialization function used for the weight of the layer
#         in_size: int
#             Input dimension of the linear layer
#         out_size: int
#             Output dimension of the linear layer
#     """

#     def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None):
#         super(linear_layer, self).__init__()

#         self.__params = locals()
#         del self.__params['__class__']
#         del self.__params['self']
#         self.in_size = in_size
#         self.out_size = out_size
#         self.bias = bias
#         self.linear = nn.Linear(in_size, out_size, bias=bias)#.to(device)
#         self.b_norm = None
#         if dropout:
#             self.dropout = nn.Dropout(p=dropout)
#         if b_norm:
#             self.b_norm = nn.BatchNorm1d(out_size)#.to(device)
#         self.init_fn = nn.init.xavier_uniform_

#         self.reset_parameters()

#     def reset_parameters(self, init_fn=None):
#         init_fn = init_fn or self.init_fn
#         if init_fn is not None:
#             init_fn(self.linear.weight, 1 / self.in_size)
#         if self.bias:
#             self.linear.bias.data.zero_()

#     def forward(self, x):
        
#         h = self.linear(x)
#         h = h.relu()
#         if self.dropout is not None:
#             h = self.dropout(h)
#         if self.b_norm is not None:
#             if h.shape[1] != self.out_size:
#                 h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
#             else:
#                 h = self.b_norm(h)
#         return h

#     # def __repr__(self):
#     #     return self.__class__.__name__ + ' (' \
#     #            + str(self.in_size) + ' -> ' \
#     #            + str(self.out_size) + ')'