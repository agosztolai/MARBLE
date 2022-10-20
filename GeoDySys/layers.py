#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import MLP, GCNConv
import torch_geometric.utils as tgu

from .lib import geometry, utils


def setup_layers(data, L, Lc, par):
    
    s, e, o = par['signal_dim'], par['emb_dim'], par['order']
    
    #diffusion
    diffusion = Diffusion(data, L, Lc)
    
    #gradient features
    grad = nn.ModuleList()
    for i in range(o):
        grad.append(AnisoConv())
        
    #cumulated number of channels after gradient features
    if par['inner_product_features']:
        cum_channels = s*o
    else:
        cum_channels = s*((1-e**(o+1))//(1-e)) - s
            
    #message passing
    convs = nn.ModuleList()
    for i in range(par['depth']):
        convs.append(GCNConv(cum_channels,cum_channels))
    
    #inner product features
    inner_products = nn.ModuleList()
    for i in range(o):
        inner_products.append(InnerProductFeatures(s, e**(i+1)))
        
    #multilayer perceptron
    mlp = MLP(in_channels=cum_channels,
              hidden_channels=par['hidden_channels'], 
              out_channels=par['out_channels'],
              num_layers=par['n_lin_layers'],
              dropout=par['dropout'],
              batch_norm=par['b_norm'],
              bias=par['bias']
              )
    
    return diffusion, grad, convs, mlp, inner_products


# =============================================================================
# Layer definitions
# =============================================================================
class AnisoConv(MessagePassing):
    """Anisotropic Convolution"""
    def __init__(self, in_channels=None, out_channels=None, **kwargs):
        super().__init__(aggr='add', **kwargs)
        
    def forward(self, x, edge_index, size, kernels=None, R=None):
        
        if R is not None:
            dim = len(kernels)
            size = (size[0]*dim, size[1]*dim)
            edge_index = expand_adjacenecy_matrix(edge_index, dim)
            
        #apply kernels
        out = []
        for K in utils.to_list(kernels):
            if R is not None:
                K = R * torch.kron(K, torch.ones(dim, dim))
                
            #transpose to change from source to target
            K = utils.to_SparseTensor(edge_index, size, value=K.t())
            out.append(self.propagate(K.t(), x=x))
            
        #[[dx1/du, dx2/du], [x1/dv, dx2/dv]] -> [dx1/du, x1/dv, dx2/du, dx2/dv]
        out = torch.stack(out, axis=2)
        out = out.reshape(out.shape[0], -1)
            
        return out

    def message_and_aggregate(self, K_t, x):
        """Message passing step. If K_t is a txs matrix (s source, t target),
           do matrix multiplication K_t@x, broadcasting over column features. 
           If K_t is a t*dimxs*dim matrix, in case of manifold computations,
           then first reshape, assuming that the columns of x are ordered as
           [dx1/du, x1/dv, ..., dx2/du, dx2/dv, ...].
           """
        n, dim = x.shape
        
        if (K_t.size(dim=1) % n*dim)==0:
            n_ch = n*dim // K_t.size(dim=1)
            x = x.view(-1, n_ch)
            
        x = K_t.matmul(x, reduce=self.aggr)
            
        return x.view(-1, dim)
    
    
def expand_adjacenecy_matrix(edge_index, dim):
    """When using rotations, we replace nodes by vector spaces so
       need to expand adjacency matrix from nxn -> n*dimxn*dim matrices"""
       
    adj = tgu.to_dense_adj(edge_index)[0]
    adj = adj.repeat_interleave(dim, dim=1).repeat_interleave(dim, dim=0)
    edge_index = tgu.sparse.dense_to_sparse(adj)[0]
    
    return edge_index
    
    
class Diffusion(nn.Module):
    """Diffusion with learned t."""

    def __init__(self, data, L, Lc=None, ic=0.0):
        super(Diffusion, self).__init__()
        
        self.L, self.Lc = L, Lc
            
        self.diffusion_time = nn.Parameter(torch.tensor(ic))
        
        
    def forward(self, x, normalise=False):
        # making sure diffusion times are positive
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)
            
        t = self.diffusion_time.detach().cpu().numpy()
        
        if self.Lc is not None:
            assert (x.shape[0]*x.shape[1] % self.Lc.shape[0])==0, \
                'Data dimension must be an integer multiple of the dimensions \
                 of the connection Laplacian!'
                 
            out = geometry.vector_diffusion(x, t, self.Lc, normalise, self.L)
                
        else:
            out = []
            for x_ in x.T:
                out.append(geometry.scalar_diffusion(x_.unsqueeze(1), t, self.L))
                
            out = torch.cat(out, axis=1)
            
        return out

    
class InnerProductFeatures(nn.Module):
    """
    Compute scaled inner-products between channel vectors.
    
    Input (V x C*D) vectors
    Output (VxC) dot products
    """

    def __init__(self, C, D):
        super(InnerProductFeatures, self).__init__()

        self.C, self.D = C, D
        
        self.O = nn.ModuleList()
        for i in range(C):
            self.O.append(nn.Linear(D, D, bias=False))
            
        self.warn = False
        
    def reset_parameters(self):
        for i in range(len(self.O)):
            self.O[i].weight.data = torch.eye(self.D)

    def forward(self, x):
        
        x = x.reshape(x.shape[0], self.D, self.C)
        
        #O_ij@x_j
        Ox = [self.O[j](x[...,j]) for j in range(self.C)]
        Ox = torch.stack(Ox, dim=2)
        
        #\sum_j x_i^T@O_ij@x_j
        xOx = torch.einsum('bki,bkj->bi', x, Ox)
        
        return xOx#torch.tanh(xOx)