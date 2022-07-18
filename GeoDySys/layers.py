#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.sparse.linalg as sla

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor
from torch_geometric.nn.dense.linear import Linear

from .utils import np2torch
from .geometry import adjacency_matrix


class AnisoConv(MessagePassing):
    """Convolution"""
    def __init__(self, in_channels, out_channels=None, lin_trnsf=True,
                 bias=False, ReLU=True, vec_norm=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        if out_channels is None:
            out_channels = in_channels
            
        if lin_trnsf:
            self.lin = Linear(in_channels, out_channels, bias=bias)
        else:
            self.lin = nn.Identity()
        
        if vec_norm:
            self.vec_norm = lambda out: F.normalize(out, p=2., dim=-1)
        else:
            self.vec_norm = nn.Identity()
        
        if ReLU:
            self.ReLU = nn.ReLU()
        else:
            self.ReLU = nn.Identity()
                
    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self, x, edge_index, K=None):
        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x) #message from all nodes to all nodes
            
        if not isinstance(K, list):
            K = [K]
            
        #evaluate all directional kernels and concatenate results columnwise
        size = (len(x[0]), len(x[1]))
        out = []
        for K_ in K:
            if K_ is not None: #anisotropic kernel
                K_ = adjacency_matrix(edge_index, size, value=K_.t())
            else: #adjacency matrix (vanilla GCN)
                K_ = adjacency_matrix(edge_index, size, value=None)
                
            out_ = self.propagate(K_.t(), x=x[0])
            out.append(out_)
            
        out = torch.cat(out, axis=1)
        out = self.lin(out)
        out = self.ReLU(out)
        out = self.vec_norm(out)
            
        return out

    def message_and_aggregate(self, K_t, x):
        #K_t is the transpose of K because of PyG convention
        return matmul(K_t, x, reduce=self.aggr)
    
    
class Diffusion(nn.Module):
    """Diffusion with learned t."""

    def __init__(self, L=None, Lc=None, ic=0.0, method='matrix_exp'):
        super(Diffusion, self).__init__()
        
        self.method = method
        self.L = L
        self.Lc = Lc
        self.diffusion_time = nn.Parameter(torch.tensor(ic))
        
        assert (L is not None) or (Lc is not None), 'No laplacian provided!'

    def forward(self, x, vector=False, normalize=False):
        
        # making sure diffusion times are positive
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)
            
        if vector:
            assert self.Lc is not None, 'Connection Laplacian is not provided!'
        if (not vector) or (normalize):
            assert self.L is not None, 'Laplacian is not provided!'
            
        t = self.diffusion_time.detach()
        if vector:
            out = compute_diffusion(x.flatten(), t, self.Lc, self.method)
            out = out.reshape(x.shape)
            if normalize:
                x_abs = x.norm(dim=-1,p=2,keepdim=True)
                out_abs = compute_diffusion(x_abs, t, self.L, self.method)
                ind = compute_diffusion(torch.ones(x.shape[0],1), t, self.L, self.method)
                out = out*out_abs/(ind*out.norm(dim=-1,p=2,keepdim=True))
        elif not vector: #diffuse componentwise
            out = []
            for i in range(x.shape[-1]):
                out.append(compute_diffusion(x[:,[i]], t, self.L, self.method))
            out = torch.cat(out, axis=1)
        else:
            NotImplementedError
            
        return out
    
    
def compute_diffusion(x, t, L, method='matrix_exp'):
    
    if method == 'matrix_exp':
        x = sla.expm_multiply(-t.numpy() * L.numpy(), x.numpy()) 
        
    return np2torch(x)