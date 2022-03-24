#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch import Tensor
from torch_sparse import matmul, SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor


class SAGE(nn.Module):
    #define all parameters in the model
    def __init__(self, in_channels, hidden_channels, num_layers, task='node'):
        super(SAGE, self).__init__()#count add (aggr='mean') to define aggregation function
        self.num_layers = num_layers
        self.task = task
        
        #initialize conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(AnisoConv(in_channels, hidden_channels, gauge=0)) #'mean','add'


    #forward computation, input is the signal and the graph
    def forward(self, x, adjs, K=None):
        
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            
            if K is not None:
                _K = SparseTensor(row=edge_index[0], col=edge_index[1], 
                                 value=K[edge_index[0],edge_index[1]],
                                 sparse_sizes=(size[0], size[1]))
                x = self.convs[i]((x, x_target), _K.t(), edge_weight=None)
            else:
                adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=None,
                        sparse_sizes=(size[0], size[1]))
                x = self.convs[i]((x, x_target), adj.t(), edge_weight=None)
                
            if i+1 != self.num_layers:
                
                # if self.graph_norm:
                #     x = x * snorm_n
                # if self.batch_norm:
                #     x = self.batchnorm_h(x)
                x = x.relu()
                # x = F.dropout(x, p=0.5, training=self.training)
                
        return x

    # for testing, we don't do minibatch
    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i+1 != self.num_layers:
                x = x.relu()
                
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
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        aggr: str = 'add',
        bias: bool = True,
        gauge = None,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_root = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, 
                             x=x, 
                             edge_weight=edge_weight,
                             size=size)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_root(x_r)

        return out
    
    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, K_transpose, x):
        return matmul(K_transpose, x[0])#, reduce=self.aggr)
        
        #we coould have a parameter to control the anisotropy like in PINConv
        
        #pass it to sum or mlp