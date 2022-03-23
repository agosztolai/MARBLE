#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_sparse import matmul, SparseTensor

from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import to_undirected, dropout_adj
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
# from torch_geometric.loader import NeighborLoader
from torch_cluster import random_walk

from cknn import cknneighbors_graph
import networkx as nx

from torch_geometric.typing import OptPairTensor


def fit_knn_graph(X, t_sample, k=10):
    
    node_feature = [list(X[i]) for i in t_sample]
    node_feature = torch.tensor(node_feature, dtype=torch.float)
    
    ckng = cknneighbors_graph(node_feature, n_neighbors=k, delta=1.0)    
    edge_index = torch.tensor(list(nx.Graph(ckng).edges), dtype=torch.int64).T
    # edge_index = knn_graph(node_feature, k=k)
    
    edge_index = to_undirected(edge_index)
    
    data = Data(x=node_feature, edge_index=edge_index)
    
    return data


def traintestsplit(data, test_size=0.1, val_size=0.5, seed=0):
    
    n = len(data.x)
    train_id, test_id = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    test_id, val_id = train_test_split(test_id, test_size=val_size, random_state=seed)
    
    train_mask = torch.zeros(n, dtype=bool)
    test_mask = torch.zeros(n, dtype=bool)
    val_mask = torch.zeros(n, dtype=bool)
    train_mask[train_id] = True
    test_mask[test_id] = True
    val_mask[val_id] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    
    return data


def train(model, data, par, writer):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if np.isscalar(par['n_neighbours']):
        par['n_neighbours'] = [par['n_neighbours'] for i in range(par['num_layers'])]
    assert len(par['n_neighbours'])==par['num_layers'], 'The number of \
    neighbours to be sampled need to be specified for all layers!'
    
    loader = NeighborSampler(data.edge_index,
                             sizes=par['n_neighbours'],
                             batch_size=par['batch_size'],
                             shuffle=True, 
                             num_nodes=data.num_nodes,
                             dropout=par['edge_dropout'],
                             )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=par['lr'])

    x = data.x.to(device)
    for epoch in range(1, par['epochs']):
        total_loss = 0
        model.train()
        
        for _, n_id, adjs in loader:
            optimizer.zero_grad() #zero gradients, otherwise accumulates gradients
            
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            if par['num_layers']==1:
                adjs = [adjs.to(device)]
            else:
                adjs = [adj.to(device) for adj in adjs]

            # compute the model for only the nodes in batch n_id
            # if hasattr(data, 'kernels'):
                # K=SparseTensor(row=n_id, col=n_id, value=data.kernels[n_id,n_id],
                #     sparse_sizes=(size[0], size[1]))
            # else:
                # K=None
            
            out = model(x[n_id], 
                        adjs,
                        data.kernels[n_id,:][:,n_id] if hasattr(data, 'kernels') else None)
            
            loss = loss_comp(out)
        
            #backprop
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * out.size(0)    
        
        total_loss /= data.num_nodes       
        writer.add_scalar("loss", total_loss, epoch)
        print("Epoch {}. Loss: {:.4f}. ".format(
                epoch, total_loss))

    return model


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


def loss_comp(out):
    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

    #loss function from word2vec
    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
    loss = -pos_loss - neg_loss
    
    return loss


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
                #     h = h * snorm_n
                # if self.batch_norm:
                #     h = self.batchnorm_h(h)
                x = x.relu()
                # x = F.dropout(x, p=0.5, training=self.training)
                
        return x

    # for testing, we don't do minibatch
    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
                
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
               
    
def model_eval(model, data):
    model.eval()
    x, edge_index = data.x, data.edge_index
    with torch.no_grad():
        out = model.full_forward(x, edge_index).cpu()

    return out