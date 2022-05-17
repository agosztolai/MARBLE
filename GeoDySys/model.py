#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP

from tensorboardX import SummaryWriter
from datetime import datetime

from .layers import AnisoConv
from .kernels import aggr_directional_derivative
from .dataloader import loaders

"""Main network"""
class net(nn.Module):
    def __init__(self, data, kernel='directional_derivative', gauge='global', **kwargs):
        super(net, self).__init__()
        
        #load default parameters
        file = os.path.dirname(__file__) + '/default_params.yaml'
        par = yaml.load(open(file,'rb'), Loader=yaml.FullLoader)
        self.par = {**par,**kwargs}
        
        #how many neighbours to sample when computing the loss function
        ncl = self.par['n_conv_layers']
        self.par['n_neighb'] = [self.par['n_neighb'] for i in range(ncl)]
        
        #channel numbers in consecutive MLP layers
        nx = data.x.shape[1]
        if kernel == 'directional_derivative':
            self.kernel = aggr_directional_derivative(data, gauge)
            nk = len(self.kernel)
            ch = [(nx*(nk**(i+1)*2**i), nx*nk**(i+1)*2**(i+1)) \
                  for i in range(ncl)]
        else: #isotropic convolutions (vanilla GCN)
            self.kernel = None
            ch = [(nx, nx) for i in range(ncl)]
        
        #initialise conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(ncl):
            self.convs.append(AnisoConv(adj_norm=self.par['adj_norm']))
        
        #initialise multilayer perceptrons
        self.MLPs = nn.ModuleList()
        for i in range(ncl-1):
            self.MLPs.append(MLP(channel_list=[ch[i][0], ch[i][1]],
                                 dropout=self.par['dropout'],
                                 batch_norm=self.par['b_norm'],
                                 bias=self.par['bias']))
        self.MLPs.append(MLP(in_channels=ch[-1][0],
                             hidden_channels=self.par['hidden_channels'], 
                             out_channels=self.par['out_channels'],
                             num_layers=self.par['n_lin_layers'],
                             dropout=self.par['dropout'],
                             batch_norm=self.par['b_norm'],
                             bias=self.par['bias']))
        
        #initialise vector normalisation
        self.vec_norm = lambda out: F.normalize(out, p=2., dim=-1)
        
    def forward(self, x, adjs, K=None):
        """Forward pass @ training (with minibatches)"""
        for i, (edge_index, _, size) in enumerate(adjs): #loop over minibatches
            #messages are passed from x_source (all nodes) to x_target
            #by convention of the NeighborSampler
            x_source = x
            x_target = x[:size[1]]
            
            x = self.convs[i]((x_source, x_target), edge_index, K=K) 
            x = self.MLPs[i](x)
            if self.par['vec_norm']:
                x = self.vec_norm(x)
                                              
        return x
    
    def evaluate(self, data):
        """Evaluate network"""
        x, edge_index = data.x, data.edge_index
            
        with torch.no_grad():
            for i in range(self.par['n_conv_layers']):
                x = self.convs[i](x, edge_index, K=self.kernel)
                x = self.MLPs[i](x)
                if self.par['vec_norm']:
                    x = self.vec_norm(x)

        return x
    
    def batch_evaluate(self, x, adjs, n_id, device):
        """Evaluate network in batches"""
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        if not isinstance(adjs, list):
            adjs = [adjs]
        adjs = [adj.to(device) for adj in adjs]
            
        #take submatrix corresponding to current batch
        if self.kernel is not None:
            K = [K_[n_id,:][:,n_id] for K_ in self.kernel]
        else:
            K = None
        
        return self(x[n_id], adjs, K)
    
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