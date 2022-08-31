#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:24:03 2022

@author: gosztola
"""

from GeoDySys import layers
import torch.nn as nn
from torch_geometric.nn import MLP

def setup_layers(data, par):
    
    diffusion = layers.Diffusion(data)
    
    #gradient features
    cum_channels = 0
    grad = nn.ModuleList()
    for i in range(par['order']):
        grad.append(layers.AnisoConv())
        cum_channels += par['signal_dim']
        cum_channels *= par['emb_dim']
        
    cum_channels += par['signal_dim']
    
    #message passing
    convs = nn.ModuleList()
    for i in range(par['depth']):
        conv = layers.AnisoConv(cum_channels, 
                                vanilla_GCN=True,
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
    inner_products = layers.InnerProductFeatures(par['signal_dim'], par['signal_dim'])
    
    #sheaf learning
    # self.sheaf = layers.SheafLearning(self.signal_dim, data.x)
    
    return diffusion, grad, convs, mlp, inner_products