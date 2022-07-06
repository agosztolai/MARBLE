#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, Linear

from tensorboardX import SummaryWriter
from datetime import datetime

from .layers import AnisoConv, Diffusion, InnerProductFeatures
from .kernels import DA, DD, gradient_op
from .dataloader import loaders
from .geometry import compute_laplacian
from .utils import parse_parameters

"""Main network"""
class net(nn.Module):
    def __init__(self, 
                 data,
                 gauge='global', 
                 root_weight=True,
                 **kwargs):
        super(net, self).__init__()
        
        self = parse_parameters(self, kwargs)
        
        #how many neighbours to sample when computing the loss function
        d = self.par['depth']
        self.par['n_neighb'] = [self.par['n_neighb'] for i in range(d)]
        
        self.L = compute_laplacian(data, k_eig=128, eps=1e-8)
        
        #kernels
        # self.kernel_DD = gradient_op(data)
        self.kernel_DD = DD(data, gauge, order=self.par['order'])
        k1 = len(self.kernel_DD)
        if not self.vanilla_GCN:
            self.kernel_DA = DA(data, gauge)
            # k2 = k1
        else: #isotropic convolutions (vanilla GCN)
            self.kernel_DA = None
            # k2 = 1
            
        #diffusion layer
        nt = self.par['n_scales']
        init = list(torch.linspace(0,self.par['large_scale'], nt))
        self.diffusion = Diffusion(data.x.shape[1], init=init)
        
        #inner pproduct features
        self.inner_products = InnerProductFeatures(nt, k1)
            
        #conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(d):
            self.convs.append(AnisoConv(adj_norm=self.par['adj_norm']))
        
        #linear layers
        out_channels = data.x.shape[1]*nt*k1#*((1-k1**(o+1))//(1-k1)-1)
        # self.lin = nn.ModuleList()
        # for i in range(d):
        #     if i < d-1:
        #         in_channels = out_channels*k2
        #         out_channels = in_channels#*2
        #         self.lin.append(Linear(in_channels, out_channels, bias = True))
        #     else:
        #         out_channels *= k2
            
        #non-linearity
        self.ReLU = nn.ReLU()
                        
        #initialise multilayer perceptron
        self.MLP = MLP(in_channels=out_channels,
                       hidden_channels=self.par['hidden_channels'], 
                       out_channels=self.par['out_channels'],
                       num_layers=self.par['n_lin_layers'],
                       dropout=self.par['dropout'],
                       batch_norm=self.par['b_norm'],
                       bias=self.par['bias'])
        
        #initialise vector normalisation
        self.vec_norm = lambda out: F.normalize(out, p=2., dim=-1)
        
        self.reset_parameters()
        
        #print settings
        print_settings(self, out_channels)
        
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
    def forward(self, x, n_id=None, adjs=None, L=None, K_DD=None, K_DA=None):
        """Forward pass. 
        Messages are passed to a set target nodes (batch variable in 
        NeighborSampler) from sources nodes. By convention, the variable x
        is constructed such that the target nodes are placed first, i.e,
        x = concat[x_target, x_other]."""
        
        x = self.diffusion(x, L)
        
        if n_id is not None:
            x = x[n_id]
        
        #taking derivatives (directional difference filters)
        if K_DD is not None:
            for i, (edge_index, _, size) in enumerate(adjs): #loop over minibatches
                x_source = x #source of messages (all nodes)
                x_target = x[:size[1]] #target of messages
                x = self.convs[i]((x_source, x_target), edge_index, K=K_DD)
                
                # x = self.inner_products(x)
                        
                x = self.MLP(x)
                if self.par['vec_norm']:
                    x = self.vec_norm(x)
            
        return x
    
    def evaluate(self, data):
        """Forward pass @ evaluation (no minibatches)"""            
        with torch.no_grad():
            adjs = [[data.edge_index, None, (data.x.shape[0], data.x.shape[0])]]
            adjs *= self.par['depth']
            return self(data.x, None, adjs, self.L, self.kernel_DD, self.kernel_DA)
    
    def batch_evaluate(self, x, adjs, n_id, device):
        """Evaluate network in batches"""
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        if (device is not None) and (adjs is not None):
            if not isinstance(adjs, list):
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]
        
        K_DD, K_DA = self.kernel_DD, self.kernel_DA
        #take submatrix corresponding to current batch  
        K_DD = [K_[n_id,:][:,n_id] for K_ in self.kernel_DD]
        if not self.vanilla_GCN:
            K_DA = [K_[n_id,:][:,n_id] for K_ in self.kernel_DA]
        else:
            K_DA = None
        
        return self(x, n_id, adjs, self.L, K_DD, K_DA)
    
    def train_model(self, data):
        """Network training"""
        
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader, val_loader, test_loader = loaders(data, 
                                                        self.par['n_neighb'], 
                                                        self.par['batch_size'])
        optimizer = torch.optim.Adam(self.parameters(), lr=self.par['lr'])
        self = self.to(device)
        x = data.x.to(device)
        
        print('\n---- Starting training ... \n')
        
        for epoch in range(1, self.par['epochs']): #loop over epochs
            
            self.train() #switch to training mode
            train_loss = 0
            for _, n_id, adjs in train_loader: #loop over batches
                optimizer.zero_grad() #zero gradients, otherwise accumulates gradients
                out = self.batch_evaluate(x, adjs, n_id, device)
                loss = loss_comp(out)
                loss.backward() #backprop
                optimizer.step()
                train_loss += float(loss)
                                
            self.eval() #switch to testing mode (this disables dropout in MLP)
            val_loss = 0
            for _, n_id, adjs in val_loader: #loop over batches                
                out = self.batch_evaluate(x, adjs, n_id, device)
                val_loss += float(loss_comp(out))  
            val_loss /= (sum(data.val_mask)/sum(data.train_mask))
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            print("Epoch: {},  Training loss: {:.4f}, Validation loss: {:.4f}"\
                  .format(epoch, train_loss, val_loss))
        
        test_loss = 0
        for _, n_id, adjs in test_loader: #loop over batches                
            out = self.batch_evaluate(x, adjs, n_id, device)
            test_loss += float(loss_comp(out))
        test_loss /= (sum(data.test_mask)/sum(data.train_mask))
        print('Final test error: {:.4f}'.format(test_loss))
    

def loss_comp(out):
    """Unsupervised loss from Hamilton et al. 2018."""
    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
    
    return -pos_loss - neg_loss


def print_settings(model, out_channels):
    
    print('---- Settings: \n')
    
    for x in model.par:
        print (x,':',model.par[x])
        
    print('\n')
    
    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('---- Number of channels to pass to the MLP: ', out_channels)
    print('---- Total number of parameters: ', np)