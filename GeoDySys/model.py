#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, Linear

from tensorboardX import SummaryWriter
from datetime import datetime

from .layers import AnisoConv
from .kernels import DA, DD
from .dataloader import loaders

"""Main network"""
class net(nn.Module):
    def __init__(self, 
                 data,
                 GCN=False,
                 gauge='global', 
                 root_weight=True,
                 **kwargs):
        super(net, self).__init__()
                
        #load default parameters
        file = os.path.dirname(__file__) + '/default_params.yaml'
        par = yaml.load(open(file,'rb'), Loader=yaml.FullLoader)
        self.par = {**par,**kwargs}
        o = self.par['order']
        nx = data.x.shape[1]
        self.GCN = GCN
        
        #how many neighbours to sample when computing the loss function
        self.par['n_neighb'] = [self.par['n_neighb'] for i in range(o)]
        
        #kernels
        if not GCN:
            self.kernel_DD = DD(data, gauge)
            nkd = len(self.kernel_DD)
        else: #isotropic convolutions (vanilla GCN)
            self.kernel_DD = None
            nkd = nka = 1
            
        #initialise conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(o):
            self.convs.append(AnisoConv(adj_norm=self.par['adj_norm']))
        
        #initialise linear layers
        d=2
        out_channels = nx*nkd*o
        self.lin = nn.ModuleList()
        for i in range(self.par['n_hops']):
            in_channels = nx*nkd*o*2**i
            out_channels = nx*nkd*o*d*2**i
            self.lin.append(Linear(in_channels, out_channels, bias = True))
        
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
        
    def reset_parameters(self):
        self.MLP.reset_parameters()
        for i in range(self.par['n_hops']):
            self.lin[i].reset_parameters()
        
    def forward(self, x, adjs=None, K=None):
        """Forward pass. 
        Messages are passed to a set target nodes (batch variable in 
        NeighborSampler) from sources nodes. By convention, the variable x
        is constructed such that the target nodes are placed first, i.e,
        x = concat[x_target, x_other]."""
        
        if K is None:
            K = self.kernel_DD
            
        out = []
        size_last = adjs[-1][-1][1] #output dim of last layer
        for i, (edge_index, _, size) in enumerate(adjs): #loop over minibatches
            x_source = x #source of messages (all nodes)
            x_target = x[:size[1]] #target of messages
            x = self.convs[i]((x_source, x_target), edge_index, K=K)
            out.append(x[:size_last])
        
        out = torch.cat(out, axis=1)
            
        out = self.MLP(out)
        if self.par['vec_norm']:
            out = self.vec_norm(out)
        
        return out
    
    def evaluate(self, data):
        """Forward pass @ evaluation (no minibatches)"""            
        with torch.no_grad():
            size = (data.x.shape[0], data.x.shape[0])
            return self(data.x, [[data.edge_index, None, size]])
    
    def batch_evaluate(self, x, adjs, n_id, device):
        """Evaluate network in batches"""
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        if (device is not None) and (adjs is not None):
            if not isinstance(adjs, list):
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]
        
        K = self.kernel_DD
        if (n_id is not None) and (not self.GCN):       
            #take submatrix corresponding to current batch  
            K = [K_[n_id,:][:,n_id] for K_ in self.kernel_DD]
        
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