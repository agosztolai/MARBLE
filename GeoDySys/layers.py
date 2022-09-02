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

from torch_householder import torch_householder_orgqr

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
        
    def forward(self, x, edge_index, kernels=None):
        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x) #message from all nodes to all nodes
            
        if self.vanilla_GCN:
             K = None
            
        if not isinstance(kernels, list):
            kernels = [kernels]
            
        #evaluate all directional kernels and concatenate results columnwise
        size = (len(x[0]), len(x[1]))
        out = []
        for K in kernels:
            if K is not None: #anisotropic kernel
                K = geometry.adjacency_matrix(edge_index, size, value=K.t())
            else: #adjacency matrix
                K = geometry.adjacency_matrix(edge_index, size, value=None)
                
            out_ = self.propagate(K.t(), x=x[0])
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

    def __init__(self, data, R=None, ic=0.0, method='matrix_exp'):
        super(Diffusion, self).__init__()
        
        self.L = geometry.compute_laplacian(data)
        self.Lc = geometry.compute_connection_laplacian(data, R)
        self.diffusion_time = nn.Parameter(torch.tensor(ic))
        self.method = method
        
    def forward(self, x, normalize=False):
        
        # making sure diffusion times are positive
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)
            
        t = self.diffusion_time.detach().numpy()
        
        if self.par['vector']:
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
    
    
class GaugeLearning():
    def __init__(self, data, X0):
        
        self.L = geometry.compute_laplacian(data).todense()
        self.X0 = X0
            
    def forward(self, id_source=None, id_target=None):
        
        L, X0 = self.L, self.X0
        if id_source is not None and id_target is not None:
            k = len(id_target)
            L = torch.tensor(self.L[:,id_source][id_target,:], dtype=torch.float64)
            X0 = torch.tensor(self.X0[id_source], dtype=torch.float64)
            
        n, p = X0[0].shape
        
        self.manifold = Stiefel(n, p=p, k=k)
        cost, euclidean_gradient = self.cost_and_derivates(L, X0)
        problem = pymanopt.Problem(
            self.manifold, cost, euclidean_gradient=euclidean_gradient
        )

        optimizer = SteepestDescent(verbosity=0)
        X = optimizer.run(problem).point
        
        # R = compute_optimal_solution(X[0].T@X[1])
        
        return X
    
    def cost_and_derivates(self, L, X0):
        euclidean_gradient = None

        @pymanopt.function.pytorch(self.manifold)
        def cost(Xi):
            return - torch.einsum('ij,jkl',L,Xi).norm() - (Xi - X0).norm()

        return cost, euclidean_gradient
    
    

    

# class SheafLearning(nn.Module):
#     def __init__(self, D, x_ic=None, orthogonal=True):
#         super(SheafLearning, self).__init__()
        
#         self.orthogonal = orthogonal
#         self.D, self.x_ic = D, x_ic
#         in_channels = 2*D
#         hidden_channels = 10
#         self.Phi = MLP(in_channels, 
#                        hidden_channels=hidden_channels,
#                        out_channels=D*D,
#                        num_layers=1,
#                        bias=False)
        
#     def reset_parameters(self):
#         self.Phi.reset_parameters()
            
#     def forward(self, x, edge_index):
        
#         x_in = torch.cat((x[edge_index[0]], x[edge_index[1]]), axis=1)
#         R_tmp = self.Phi(x_in)
#         R_tmp = R_tmp.reshape(-1, self.D, self.D)
        
#         n = x.shape[0]
#         if self.orthogonal:
#             hh = R_tmp.tril(diagonal=-1) + torch.eye(self.D).unsqueeze(0).repeat(len(x_in),1,1)
#             R_tmp = torch_householder_orgqr(hh)
#             R_tmp = R_tmp.reshape(-1, self.D, self.D)
         
#         R = torch.empty(n, n, self.D, self.D)
#         R[edge_index[0], edge_index[1], :,:] = R_tmp
        
#         return R

    
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