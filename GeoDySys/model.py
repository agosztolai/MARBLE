#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP

from tensorboardX import SummaryWriter
from datetime import datetime

from .layers import AnisoConv, Diffusion
from .kernels import DD, gradient_op
from .dataloader import loaders
from .geometry import compute_laplacian, compute_tangent_frames, compute_connection_laplacian
from .utils import parse_parameters, print_settings, check_parameters

"""Main network"""
class net(nn.Module):
    def __init__(self, 
                 data,
                 local_gauge=False, 
                 include_identity=False,
                 **kwargs):
        super(net, self).__init__()
        
        self.par = parse_parameters(self, data, kwargs)
        self.par = check_parameters(self.par, data)
        self.include_identity = include_identity
        L = compute_laplacian(data)
        
        if local_gauge:
            local_gauge, R = \
                compute_tangent_frames(
                    data, 
                    n_geodesic_nb=self.par['n_geodesic_nb'],
                    return_predecessors=True
                    )
            Lc = compute_connection_laplacian(L, R)
        else:
            local_gauge=None
        
        #kernels
        # self.kernel = gradient_op(data)
        self.kernel = DD(data, local_gauge=local_gauge)
            
        #diffusion layer
        nt = self.par['n_scales']
        init = list(torch.linspace(0,self.par['large_scale'], nt))
        self.diffusion = Diffusion(L, data.x.shape[1], init=init)
            
        #conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        in_channels = data.x.shape[1]*nt
        cum_channels = 0
        for i in range(self.par['order']):
            in_channels *= len(self.kernel)
            if include_identity:
                in_channels += 1
            self.convs.append(AnisoConv(in_channels, 
                                        vec_norm=self.par['vec_norm']
                                        )
                              )
            cum_channels += in_channels
            
        #multilayer perceptron
        self.MLP = MLP(in_channels=cum_channels,
                       hidden_channels=self.par['hidden_channels'], 
                       out_channels=self.par['out_channels'],
                       num_layers=self.par['n_lin_layers'],
                       dropout=self.par['dropout'],
                       batch_norm=self.par['b_norm'],
                       bias=self.par['bias']
                       )
        
        self.reset_parameters()
        
        print_settings(self, in_channels)
        
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
    def forward(self, x, n_id=None, adjs=None, K=None):
        """Forward pass. 
        Messages are passed to a set target nodes (batch variable in 
        NeighborSampler) from sources nodes. By convention, the variable x
        is constructed such that the target nodes are placed first, i.e,
        x = concat[x_target, x_other]."""
        
        x = self.diffusion(x)
        
        x = x[n_id] if n_id is not None else x  
        out = [x] if self.include_identity else []
            
        #taking derivatives (directional difference filters)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_source = x #all nodes
            x_target = x[:size[1]]
            x = self.convs[i]((x_source, x_target), edge_index, K=K)
            out.append(x)
            
        out = [o[:size[1]] for o in out] #only take target nodes
        x = torch.cat(out, axis=1)
        
        return self.MLP(x)
    
    def evaluate(self, data):
        """Forward pass @ evaluation (no minibatches)"""            
        with torch.no_grad():
            adjs = [[data.edge_index, None, (data.x.shape[0], data.x.shape[0])]]
            adjs *= self.par['order']
            return self(data.x, None, adjs, self.kernel)
    
    def batch_evaluate(self, x, adjs, n_id, device):
        """Evaluate network in batches"""
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        if (device is not None) and (adjs is not None):
            if not isinstance(adjs, list):
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]
        
        #take submatrix corresponding to current batch  
        K_DD = [K_[n_id,:][:,n_id] for K_ in self.kernel]
        
        return self(x, n_id, adjs, K_DD)
    
    def train_model(self, data):
        """Network training"""
        
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader, val_loader, test_loader = loaders(data, self.par)
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
    """Unsupervised loss from GraphSAGE (Hamilton et al. 2018.)"""
    batch, pos_batch, neg_batch = out.split(out.size(0) // 3, dim=0)
    pos_loss = F.logsigmoid((batch * pos_batch).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(batch * neg_batch).sum(-1)).mean()
    
    return -pos_loss - neg_loss