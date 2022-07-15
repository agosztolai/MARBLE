#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.sparse.linalg as sla

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor
from torch_geometric.nn.dense.linear import Linear

from .utils import adjacency_matrix, np2torch


"""Convolution"""
class AnisoConv(MessagePassing):    
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
        if isinstance(x, Tensor):
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
    """Applies diffusion with learned t."""

    def __init__(self, L, C_inout, method='matrix_exp', init=[0]):
        super(Diffusion, self).__init__()
        
        self.C_inout = C_inout
        self.method = method
        self.L = L
        self.diffusion_time = []
        for i in init:
            self.diffusion_time.append(nn.Parameter(torch.Tensor(i)))

    def forward(self, x):
        
        # making sure diffusion times are positive
        with torch.no_grad():
            for d in self.diffusion_time:
                d.data = torch.clamp(d, min=1e-8)

        assert x.shape[-1] == self.C_inout, \
            "x has wrong shape {}. Last dim should be {}".format(x.shape, self.C_inout)
            
        if self.method == 'matrix_exp':
            
            out = []
            for t in self.diffusion_time:
                t = t.detach()
                for i in range(x.shape[-1]):
                    x_diff = sla.expm_multiply(-t.numpy() * self.L.numpy(), x[:,[i]].numpy()) 
                    out.append(np2torch(x_diff))
                    
            out = torch.cat(out, axis=1)

        else:
            NotImplementedError
            
        return out