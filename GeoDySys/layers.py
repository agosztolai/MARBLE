#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.sparse.linalg as sla

import torch
from torch import Tensor
import torch.nn as nn

from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor

from .utils import adjacency_matrix, np2torch


"""Convolution"""
class AnisoConv(MessagePassing):    
    def __init__(self, 
                 adj_norm=False, 
                 root_weight=True,
                 eps=0.,
                 **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.adj_norm = adj_norm
        self.root_weight = root_weight
        
        #uf root_weight=True then eps is a trainable parameter
        if root_weight:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        self.reset_parameters(eps)

    def reset_parameters(self, eps):
        self.eps.data.fill_(eps)

    def forward(self, x, edge_index, K=None):
        """Forward pass"""
        if isinstance(x, Tensor):
            #when there is no minibatching, messages are passed from all source
            #to all target nodes
            x: OptPairTensor = (x, x)
            
        if not isinstance(K, list):
            K = [K]
            
        size = (len(x[0]), len(x[1]))
        out = []
        #evaluate all directional kernels and concatenate results columnwise
        for K_ in K:
            if K_ is not None: #anisotropic kernel
                K_ = adjacency_matrix(edge_index, size, value=K_.t())
            else: #adjacency matrix (vanilla GCN)
                K_ = adjacency_matrix(edge_index, size, value=None)
                
            out_ = self.propagate(K_.t(), x=x[0])
                
            #skip connection
            out_ += self.eps * x[1] 
                
            if self.adj_norm: #adjacency features
                adj = adjacency_matrix(edge_index, size)
                out_ = adj_norm(x[0], out_, adj.t(), K_.t(), float(self.eps))
                    
            out.append(out_)
                    
        out = torch.cat(out, axis=1)
  
        return out

    def message_and_aggregate(self, K_t, x):
        """Anisotropic convolution step. Need to be transposed because of PyG 
        convention. This is executed if input to propagate() is a SparseTensor"""
        return matmul(K_t, x, reduce=self.aggr)
    
    
class Diffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.
    In the spectral domain this becomes 
        f_out = e^(lambda_i t) f_in
    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal
      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values 
    """

    def __init__(self, C_inout, method='matrix_exp', init=[0]):
        super(Diffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = []
        for i in init:
            self.diffusion_time.append(nn.Parameter(torch.Tensor(C_inout)))
        self.method = method # one of ['matrix_exp', 'spectral', 'implicit_dense']
        
    def reset_parameters(self):
        for d in self.diffusion_time:
            nn.init.constant_(d, 0.0)   

    def forward(self, x, L):

        L, mass, evals, evecs = L
        
        if mass is None:
            mass = torch.tensor(1)
        
        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            for d in self.diffusion_time:
                d.data = torch.clamp(d, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))
            
        if self.method == 'matrix_exp':
            
            out = []
            for t in self.diffusion_time:
                t = t.detach()
                t = t.unsqueeze(-1) if t.nelement() == 1 else t
                for i, t in enumerate(t):
                    x_diff = sla.expm_multiply(-t.numpy() * L.numpy(), x[:,[i]].numpy()) 
                    out.append(np2torch(x_diff))
                    
            return torch.cat(out, axis=1)

        # elif self.method == 'spectral':

        #     # Transform to spectral
        #     x_spec = evecs.T@x*mass.unsqueeze(-1)

        #     # Diffuse
        #     time = self.diffusion_time
        #     diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
        #     x_spec *= diffusion_coefs

        #     # Transform back to per-vertex 
        #     return evecs@x_spec
            
        # elif self.method == 'implicit_dense':
        #     V = x.shape[-2]

        #     # Form the dense matrices (M + tL) with dims (B,C,V,V)
        #     mat_dense = L.unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
        #     mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        #     mat_dense += torch.diag_embed(mass).unsqueeze(1)

        #     # Factor the system
        #     cholesky_factors = torch.linalg.cholesky(mat_dense)
            
        #     # Solve the system
        #     rhs = x * mass.unsqueeze(-1)
        #     rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
        #     sols = torch.cholesky_solve(rhsT, cholesky_factors)
        #     return torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method") 

    
def adj_norm(x, out, adj_t, K_t, eps):
    """Normalize features by mean of neighbours"""
    ones = torch.ones([x.shape[0],1])
    # x = x.norm(dim=-1,p=2, keepdim=True)
    mu_x = (matmul(adj_t, x) + eps*out) / (matmul(adj_t, ones) + (eps>0)*1)
    K1 = matmul(K_t, ones)
    # sigma_x = (matmul(adj_t, x**2) / matmul(adj_t, ones)) - mu_x**2
    out -= (K1*mu_x)#.repeat([1,out.shape[1]//x.shape[1]])
    
    return out