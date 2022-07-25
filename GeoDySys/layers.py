#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal

from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MLP

from GeoDySys import geometry


class AnisoConv(MessagePassing):
    """Convolution"""
    def __init__(self, in_channels=None, out_channels=None, lin_trnsf=False,
                 bias=False, ReLU=False, vec_norm=False, 
                 vanilla_GCN=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.vanilla_GCN = vanilla_GCN
    
        if lin_trnsf:
            if out_channels is None:
                out_channels = in_channels
            self.lin = Linear(in_channels, out_channels, bias=bias)   
        else:
            self.lin = nn.Identity()
        
        if vec_norm:
            self.vec_norm = lambda x: F.normalize(x, p=2., dim=-1)
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
            
        if self.vanilla_GCN:
             K = None
            
        if not isinstance(K, list):
            K = [K]
            
        #evaluate all directional kernels and concatenate results columnwise
        size = (len(x[0]), len(x[1]))
        out = []
        for K_ in K:
            if K_ is not None: #anisotropic kernel
                K_ = geometry.adjacency_matrix(edge_index, size, value=K_.t())
            else: #adjacency matrix
                K_ = geometry.adjacency_matrix(edge_index, size, value=None)
                
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
        self.L, self.Lc = L, Lc
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
            
        t = self.diffusion_time.detach().numpy()
        if vector:
            out = geometry.compute_diffusion(x.flatten(), t, self.Lc, self.method)
            out = out.reshape(x.shape)
            if normalize:
                x_abs = x.norm(dim=-1,p=2,keepdim=True)
                out_abs = geometry.compute_diffusion(x_abs, t, self.L, self.method)
                ind = geometry.compute_diffusion(torch.ones(x.shape[0],1), t, self.L, self.method)
                out = out*out_abs/(ind*out.norm(dim=-1,p=2,keepdim=True))
        else: #diffuse componentwise
            out = []
            for i in range(x.shape[-1]):
                out.append(geometry.compute_diffusion(x[:,[i]], t, self.L, self.method))
            out = torch.cat(out, axis=1)
            
        return out
    

class SheafLearning(nn.Module):
    def __init__(self, D, x_ic=None, sym=True):
        super(SheafLearning, self).__init__()
        
        self.D, self.x_ic, self.sym = D, x_ic, sym
        in_channels = D if sym else 2*D
        hidden_channels = 10
        self.Phi = MLP(in_channels, 
                       hidden_channels=hidden_channels,
                       out_channels=D*D,
                       num_layers=1,
                       bias=False)
        
    def reset_parameters(self):
        self.Phi.reset_parameters()
            
    def forward(self, x, edge_index):
        
        if self.sym:
            x_in = (x[edge_index[0]] - x[edge_index[1]]).abs()
        else:
            x_in = torch.cat((x[edge_index[0]], x[edge_index[1]]), axis=1)
            
        R_tmp = self.Phi(x_in)
        R_tmp = R_tmp.reshape(-1, self.D, self.D)
        
        n = x.shape[0]
        R = torch.empty(n,n,self.D,self.D)
        R[edge_index[0], edge_index[1], :,:] = R_tmp
        
        return R

    
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