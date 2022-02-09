#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, dropout_adj
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
# from torch_geometric.loader import NeighborLoader
from torch_cluster import random_walk

from torch_sparse import SparseTensor, matmul

from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from cknn import cknneighbors_graph
import networkx as nx


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
    
    train_id, test_id = train_test_split(np.arange(data.x.shape[0]), test_size=test_size, random_state=seed)
    test_id, val_id = train_test_split(test_id, test_size=val_size, random_state=seed)
    
    train_mask = torch.zeros(len(data.x), dtype=bool)
    test_mask = torch.zeros(len(data.x), dtype=bool)
    val_mask = torch.zeros(len(data.x), dtype=bool)
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
    
    assert len(par['n_neighbours'])==par['num_layers'], 'The number of \
    neighbours to be sampled need to be specified for all layers!'
    
    loader = NeighborSampler(data.edge_index,
                             sizes=par['n_neighbours'],
                             batch_size=par['batch_size'],
                             shuffle=True, 
                             num_nodes=data.num_nodes,
                             dropout=0.3,
                             )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=par['lr'])

    x, edge_index = data.x.to(device), data.edge_index.to(device)
    for epoch in range(1, par['epochs']):
        total_loss = 0
        model.train()
        
        for _, n_id, adjs in loader:
            optimizer.zero_grad() #zero gradients, otherwise accumulates gradients
            
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]

            # compute the model for only the nodes in batch n_id
            out = model(x[n_id], adjs)
            
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


class NeighborSampler_dropout(RawNeighborSampler):
    def __init__(self,*args,dropout=0.1,**kwargs):
        super().__init__(*args,**kwargs)
        self.dropout=dropout


class NeighborSampler(NeighborSampler_dropout):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()
        edge_index = torch.stack((row,col))
        edge_index, _ = dropout_adj(edge_index, p=self.dropout)
        row = edge_index[0]
        col = edge_index[1]

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
        
        #initialize SAGEConv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels)) #'mean','add'

    #forward computation, input is the signal and the graph
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)               
            
        if self.task == 'graph':
            #do here a hiararchical pool (check package)
            x = pyg_nn.global_mean_pool(x, batch)
                
        return x

    # for testing, we don't do minibatch training
    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
                
        return x
    
    
class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
    
    
def model_eval(model, data):
    model.eval()
    x, edge_index = data.x, data.edge_index
    with torch.no_grad():
        out = model.full_forward(x, edge_index).cpu()

    return out