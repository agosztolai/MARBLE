#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MLP
import torch_geometric.utils as tgu

from .lib import geometry, utils


def setup_layers(data, R, par):
    
    #diffusion
    diffusion = Diffusion(data, R)
    
    #gradient features
    cum_channels = 0
    grad = nn.ModuleList()
    for i in range(par['order']):
        grad.append(AnisoConv())
        cum_channels += par['signal_dim']
        cum_channels *= par['emb_dim']
            
    #message passing
    convs = nn.ModuleList()
    for i in range(par['depth']):
        conv = AnisoConv(cum_channels, 
                         lin_trnsf=True,
                         ReLU=True,
                         vec_norm=par['vec_norm'],
                         )
        convs.append(conv)
        
    #multilayer perceptron
    mlp = MLP(in_channels=cum_channels,
              hidden_channels=par['hidden_channels'], 
              out_channels=par['out_channels'],
              num_layers=par['n_lin_layers'],
              dropout=par['dropout'],
              batch_norm=par['b_norm'],
              bias=par['bias']
              )
    
    #inner product features
    inner_products = InnerProductFeatures(par['signal_dim'], par['signal_dim'])
    
    return diffusion, grad, convs, mlp, inner_products


class AnisoConv(MessagePassing):
    """Convolution"""
    def __init__(self, in_channels=None, out_channels=None, lin_trnsf=False,
                 bias=False, ReLU=False, vec_norm=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
            
        self.lin = self.vec_norm = self.ReLU = nn.Identity()
        if lin_trnsf:
            self.lin = Linear(in_channels, out_channels, bias=bias)   
        
        if vec_norm:
            self.vec_norm = lambda x: F.normalize(x, p=2., dim=-1)
        
        if ReLU:
            self.ReLU = nn.ReLU()
                            
    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self, x, edge_index, size, kernels=None, R=None):
           
        dim = x.shape[1]
        kernels = utils.to_list(kernels)
        
        #when using rotations, we replace nodes by vector spaces so
        #need to expand nxn -> n*dimxn*dim matrices
        if R is not None:
            size = (size[0]*dim, size[1]*dim)
            adj = tgu.to_dense_adj(edge_index)[0]
            adj = adj.repeat_interleave(dim,dim=1).repeat_interleave(dim,dim=0)
            edge_index = tgu.sparse.dense_to_sparse(adj)[0]
            R = R.swapaxes(1,2).reshape(size[0], size[0]) #make block matrix
            
        #apply kernels
        out = []
        for K in kernels:
            if R is not None:
                K = torch.kron(K, torch.ones(dim, dim))
                K *= R
                
            #transpose to change from source to target
            K = utils.to_SparseTensor(edge_index, size, value=K.t())
            
            out_ = self.propagate(K.t(), x=x)
            out.append(out_)
            
        out = torch.cat(out, axis=1) #concatenate columnwise
        
        out = self.lin(out)
        out = self.ReLU(out)
        out = self.vec_norm(out)
            
        return out

    def message_and_aggregate(self, K_t, x):
        n, dim = x.shape
                
        if K_t.size(dim=1) == n*dim:
            x = x.view(-1, 1)
            
        x = K_t.matmul(x, reduce=self.aggr)
            
        return x.view(-1, dim)
    
    
class Diffusion(nn.Module):
    """Diffusion with learned t."""

    def __init__(self, data, R=None, ic=0.0):
        super(Diffusion, self).__init__()
        
        self.L = geometry.compute_laplacian(data)
        if R is not None:
            self.Lc = geometry.compute_connection_laplacian(data, R)
            self.vector = True
        else:
            self.vector = False
        self.diffusion_time = nn.Parameter(torch.tensor(ic))
        
    def forward(self, x, normalize=False):
        
        # making sure diffusion times are positive
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)
            
        t = self.diffusion_time.detach().numpy()
        
        if self.vector:
            out = geometry.compute_diffusion(x.flatten(), t, self.Lc)
            out = out.reshape(x.shape)
            if normalize:
                x_abs = x.norm(dim=-1,p=2,keepdim=True)
                out_abs = geometry.compute_diffusion(x_abs, t, self.L)
                ind = geometry.compute_diffusion(torch.ones(x.shape[0],1), t, self.L)
                out = out*out_abs/(ind*out.norm(dim=-1,p=2,keepdim=True))
        else: #diffuse componentwise
            out = []
            for i in range(x.shape[-1]):
                out.append(geometry.compute_diffusion(x[:,[i]], t, self.L))
            out = torch.cat(out, axis=1)
            
        return out

    
class InnerProductFeatures(nn.Module):
    """
    Compute scaled inner-products between channel vectors.
    
    Input:
        - vectors: (V,C*D)
    Output:
        - dots: (V,C)
    """

    def __init__(self, C, D):
        super(InnerProductFeatures, self).__init__()

        self.C, self.D = C, D

        self.O = []
        for i in range(C):
            self.O.append(orthogonal(nn.Linear(D, D, bias=False)))
            
        self.warn = False
            
    def reset_parameters(self):
        for lin in self.O:
            lin.reset_parameters()

    def forward(self, x):
        
        if self.C==1:
            if not self.warn:
                print('There is only one channel so cannot take inner products! \
                      Taking magnitude instead!')
                self.warn = True
            return x.norm(dim=1, p=2, keepdim=True)
        
        with torch.no_grad():
            for O in self.O:
                O.weight.data = O.weight.data.clamp(min=1e-8)
        
        x = x.reshape(x.shape[0], self.D, self.C)
        x = x.swapaxes(1,2) #make D the last dimension

        Ox = []
        for i in range(self.C): #batch over features
            Ox.append(self.O[i](x[:,i,:])) #broadcast over vertices  

        Ox = torch.stack(Ox, dim=1)
        Ox = Ox.unsqueeze(1).repeat(1,self.C,1,1)
        x = x.unsqueeze(1).repeat(1,self.C,1,1)
        x = x.swapaxes(1,2) #transpose 

        return (x*Ox).sum(2).sum(-1)#torch.tanh(dots)