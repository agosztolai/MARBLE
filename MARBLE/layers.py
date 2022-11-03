#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.functional import normalize

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import MLP
import torch_geometric.utils as tgu

from .lib import geometry as g
from .lib import utils


def setup_layers(model):
    
    par = model.par
    
    s, e, o = par['dim_signal'], par['dim_emb'], par['order']
    
    #diffusion
    diffusion = Diffusion(model.L, model.Lc)
    
    #gradient features
    grad = nn.ModuleList(AnisoConv(par['vec_norm']) for i in range(o))
        
    #cumulated number of channels after gradient features
    cum_channels = s*((1-e**(o+1))//(1-e))
    if par['inner_product_features']:
        cum_channels //= s
        if s==1:
            cum_channels = o+1
    
    #inner product features
    ip = InnerProductFeatures(cum_channels, s)
    
    #encoder
    channel_list = [cum_channels] + \
                    (par['n_lin_layers']-1) * [par['hidden_channels']] + \
                    [par['out_channels']]
    if par['pretrained']:
        enc = autoencoder(channel_list)
    else:
        enc = MLP(channel_list=channel_list,
                  dropout=par['dropout'],
                  norm=par['batch_norm'],
                  bias=par['bias']
                  )
    
    return diffusion, grad, enc, ip


# =============================================================================
# Layer definitions
# =============================================================================
class Diffusion(nn.Module):
    """Diffusion with learned t."""

    def __init__(self, L, Lc=None, tau0=0.0):
        super().__init__()
                
        self.diffusion_time = nn.Parameter(torch.tensor(float(tau0)))
        self.par = {'L': L, 'Lc': Lc}
        
        if Lc is None:
            self.par['evals'], self.par['evecs'] = g.compute_eigendecomposition(L)
        else:
            self.par['evals_L'], self.par['evecs_L'] = g.compute_eigendecomposition(L)
            self.par['evals_Lc'], self.par['evecs_Lc'] = g.compute_eigendecomposition(Lc)
        
    def forward(self, x, method='spectral', normalise=False):
        
        # making sure diffusion times are positive
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)
            
        t = self.diffusion_time
        
        if self.par['Lc'] is not None:
            out = g.vector_diffusion(x, t, method, self.par, normalise)
        else:
            out = [g.scalar_diffusion(x_, t, method, self.par) for x_ in x.T]
            out = torch.cat(out, axis=1)
            
        return out
    
    
class AnisoConv(MessagePassing):
    """Anisotropic Convolution"""
    def __init__(self, vec_norm=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.vec_norm = vec_norm
        
    def forward(self, x, edge_index, size, kernels=None, R=None):
        
        if R is not None:
            n, _, _, dim = R.shape
            R = R.swapaxes(1,2).reshape(n*dim, n*dim) #make block matrix
            size = (size[0]*dim, size[1]*dim)
            edge_index = expand_adjacenecy_matrix(edge_index, dim)
            
        #apply kernels
        out = []
        for K in utils.to_list(kernels):
            if R is not None:
                K = R * torch.kron(K, torch.ones(dim, dim).to(x.device))
                
            #transpose to change from source to target
            K = utils.to_SparseTensor(edge_index, size, value=K.t())
            out.append(self.propagate(K.t(), x=x))
            
        #[[dx1/du, dx2/du], [dx1/dv, dx2/dv]] -> [dx1/du, dx1/dv, dx2/du, dx2/dv]
        out = torch.stack(out, axis=2)
        out = out.view(out.shape[0], -1)
        
        if self.vec_norm:
            out = normalize(out, dim=-1, p=2)
            
        return out

    def message_and_aggregate(self, K_t, x):
        """Message passing step. If K_t is a txs matrix (s sources, t targets),
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
    
    
class autoencoder(nn.Module):
    """Autoencoder to initialise weights"""
    def __init__(self, channel_list):
        super().__init__()
         
        self.encoder = MLP(channel_list=channel_list)
        self.decoder = MLP(channel_list=channel_list[::-1])
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    
class InnerProductFeatures(nn.Module):
    """
    Compute scaled inner-products between channel vectors.
    
    Input: (V x C*D) vector of (V x n_i) list of vectors with \sum_in_i = C*D
    Output: (VxC) dot products
    """

    def __init__(self, C, D):
        super().__init__()

        self.C, self.D = C, D
        
        self.O = nn.ModuleList()
        for i in range(C):
            self.O.append(nn.Linear(D, D, bias=False))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for i in range(len(self.O)):
            self.O[i].weight.data = torch.eye(self.D)

    def forward(self, x):
        
        if not isinstance(x, list):
            x = [x]
            
        x = [x_.reshape(x_.shape[0], self.D, -1) for x_ in x]
            
        #for scalar signals take magnitude
        if self.D==1:
            x = [x_.norm(dim=2) for x_ in x]
            
            return torch.cat(x, axis=1)
        
        #for vector signals take inner products
        else:  
            x = torch.cat(x, axis=2)
            
            assert x.shape[2]==self.C, 'Number of channels is incorrect!'
            
            #O_ij@x_j
            Ox = [self.O[j](x[...,j]) for j in range(self.C)]
            Ox = torch.stack(Ox, dim=2)
            
            #\sum_j x_i^T@O_ij@x_j
            xOx = torch.einsum('bki,bkj->bi', x, Ox)
            
            return torch.tanh(xOx).reshape(x.shape[0], -1)