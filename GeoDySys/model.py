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
    def __init__(self, in_channels, hidden_channels, out_channels=None, **kwargs):
        super(net, self).__init__()#count add (aggr='mean') to define aggregation function
        
        self.n_layers = kwargs['n_layers'] if 'n_layers' in kwargs else 1
        
        #initialize conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(self.n_layers):
            self.convs.append(AnisoConv(
                adj_norm=kwargs['adj_norm'] if 'adj_norm' in kwargs else False))
            
        self.MLP = MLP(in_channels,
                       hidden_channels=hidden_channels, 
                       out_channels=out_channels,
                       n_lin_layers=kwargs['n_lin_layers'] if 'n_lin_layers' in kwargs else 1,
                       activation=kwargs['activation'] if 'activation' in kwargs else False,
                       dropout=kwargs['dropout'] if 'dropout' in kwargs else 0.,
                       b_norm=kwargs['b_norm'] if 'b_norm' in kwargs else False)
        
    #forward computation, input is the signal and the graph
    def forward(self, x, adjs, K=None):
        
        for i, (edge_index, _, size) in enumerate(adjs):
            x_source = x
            x_target = x[:size[1]]  # Target nodes first
            
            #perform convolution
            if K is not None: #anisotropic kernel            
                #evaluate all directional kernels and concatenate results columnwise
                x = self.convs[i]((x_source, x_target), edge_index, K=K, edge_weight=None, size=size)
                
            else: #use adjacency matrix (standard GCN)
                x = self.convs[i]((x_source, x_target), edge_index, edge_weight=None, size=size)
                
            x = self.MLP(x)
                                              
        return x

    # for testing, we don't do minibatch
    def full_forward(self, x, edge_index, K=None):        
        for conv in self.convs:
            x = conv(x, edge_index, K=K, size=(x.shape[0],x.shape[0]))
            x = self.MLP(x)
            
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
    def __init__(self, adj_norm=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.adj_norm = adj_norm

    def forward(self, x, edge_index, K=None, edge_weight=None, size=None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
            
        adj = self.adjacency_matrix(edge_index, size)
        if K is not None: #use custom kernel if available
            out=[]
            for K_ in K:
                K_ = SparseTensor(row=edge_index[0], col=edge_index[1], 
                                  value=K_[edge_index[0],edge_index[1]],
                                  sparse_sizes=(size[0], size[1]))
                out.append(self.propagate(K_.t(), x=x, edge_weight=edge_weight, size=size))
            out = torch.cat(out, axis=1)
        else:
            out = self.propagate(adj.t(), x=x, edge_weight=edge_weight, size=size)
        
        #normalize each aggregated value by the mean of neighbours
        if self.adj_norm:
            out = self.adj_norm_(x[0], out, adj.t())

        x_r = x[1]
        if x_r is not None:
            # out += self.lin_r(x_r)
            out += x_r
            
        return out
    
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

    def message_and_aggregate(self, K_t, x):
        return matmul(K_t, x[0], reduce=self.aggr)
    
    # def message(self, x_j, edge_weight):
    #     return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    
class MLP(nn.Module):
    """
    Multi-layer perceptron composed of linear layers
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 n_lin_layers=1,
                 activation=False,
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
            
        self.lin = nn.ModuleList()
        for _ in range(n_lin_layers - 1):
            self.lin.append(Linear(in_channels, hidden_channels, bias=True))
        
        #final layer
        self.lin.append(Linear(hidden_channels, out_channels, bias=False))
        
        self.activation = nn.ReLU() if activation else False
        self.dropout = nn.Dropout(dropout)
        self.b_norm = (lambda out: F.normalize(out, p=2., dim=-1)) if b_norm else False
        
        self.init_fn = nn.init.xavier_uniform_
        self.reset_parameters()
        self.n_lin_layers = n_lin_layers
        
    def reset_parameters(self):
        for l in self.lin:
            l.reset_parameters()

    def forward(self, x):
        for i in range(self.n_lin_layers):
            x = self.lin[i](x)
            if self.activation and i+1!=self.n_layers:
                x = self.activation(x)
            x = self.dropout(x)
            if self.b_norm:
                x = self.b_norm(x)
        return x
               

#         init_fn: callable, optional
#             Initialization function to use for the weight of the layer. Default is
#             :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
#             (Default value = None)


#         self.__params = locals()
#         del self.__params['__class__']
#         del self.__params['self']

#     def reset_parameters(self, init_fn=None):
#         init_fn = init_fn or self.init_fn
#         if init_fn is not None:
#             init_fn(self.linear.weight, 1 / self.in_size)
#         if self.bias:
#             self.linear.bias.data.zero_()

#     def forward(self, x):
        
#         if self.dropout is not None:
#             h = self.dropout(h)
#         if self.b_norm is not None:
#             if h.shape[1] != self.out_size:
#                 h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
#             else:
#                 h = self.b_norm(h)
#         return h