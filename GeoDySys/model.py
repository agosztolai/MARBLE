#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from tensorboardX import SummaryWriter
from datetime import datetime
from .dataloader import NeighborSampler
from .layers import AnisoConv, MLP
from .kernels import aggr_directional_derivative
import yaml
import os
import numpy as np

"""Main network architecture"""
class net(nn.Module):
    def __init__(self, data, kernel='directional_derivative', gauge='global', **kwargs):
        super(net, self).__init__()
        
        #load default parameters
        file = os.path.dirname(__file__) + '/default_params.yaml'
        par = yaml.load(open(file,'rb'), Loader=yaml.FullLoader)
        self.par = {**par,**kwargs}
        
        if kernel == 'directional_derivative':
            self.kernel = aggr_directional_derivative(data, gauge)
            in_channels = len(self.kernel)*data.x.shape[1]
        else: #isotropic convolutions (vanilla GCN)
            self.kernel = None
            in_channels = data.x.shape[1]
        
        #initialize conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(self.par['n_conv_layers']):
            self.convs.append(AnisoConv(adj_norm=self.par['adj_norm']))
        
        #initialize multilayer perceptron
        self.MLP = MLP(in_channels=in_channels*2**(self.par['n_conv_layers']-1),
                       hidden_channels=self.par['hidden_channels'], 
                       out_channels=self.par['out_channels'],
                       n_lin_layers=self.par['n_lin_layers'],
                       activation=self.par['activation'],
                       dropout=self.par['dropout'],
                       vec_norm=self.par['vec_norm'])
        
    def forward(self, x, adjs, K=None):
        """Forward pass @ training (with minibatches)"""
        for i, (edge_index, _, size) in enumerate(adjs): #loop over minibatches
            #messages are passed from x_source (all nodes) to x_target
            #by convention of the NeighborSampler
            x_source = x
            x_target = x[:size[1]]
            
            x = self.convs[i]((x_source, x_target), edge_index, K=K, size=size)
            
        x = self.MLP(x)
                                              
        return x

    def forward_test(self, x, edge_index, K=None):  
        """Forward pass @ testing (no minibatches)"""
        for conv in self.convs:
            x = conv(x, edge_index, K=K, size=(x.shape[0],x.shape[0]))
            
        x = self.MLP(x)
            
        return x        
    
    def train_model(self, data):
        """Network training"""
        
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self = self.to(device)
        
        if np.isscalar(self.par['n_neighbours']):
            n_neighbours = [self.par['n_neighbours'] for i in range(self.par['n_conv_layers'])]
        
        train_loader = NeighborSampler(data.edge_index,
                                 sizes=n_neighbours,
                                 batch_size=self.par['batch_size'],
                                 shuffle=True, 
                                 num_nodes=data.num_nodes,
                                 dropout=self.par['edge_dropout'],
                                 )
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.par['lr'])

        x = data.x.to(device)
        for epoch in range(1, self.par['epochs']): #loop over epochs
            
            self.train()
            train_loss = 0
            for _, n_id, adjs in train_loader: #loop over batches
                optimizer.zero_grad() #zero gradients, otherwise accumulates gradients
                
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                if not isinstance(adjs, list):
                    adjs = [adjs]
                adjs = [adj.to(device) for adj in adjs]
                    
                #take submatrix corresponding to current batch
                if self.kernel is not None:
                    K = [K_[n_id,:][:,n_id] for K_ in self.kernel]
                else:
                    K = None
                
                out = self(x[n_id], adjs, K)
                loss = loss_comp(out)
                loss.backward() #backprop
                optimizer.step()

                train_loss += float(loss) * out.size(0)
                
            self.eval()
            # out = self.eval_model(data, test=True)
            # test_loss = loss_comp(out) / data.num_nodes
            test_loss = 0
            
            train_loss /= data.num_nodes
            writer.add_scalar("loss", train_loss, epoch)
            print("Epoch {}  Training loss: {:.4f} Test loss: {:.4f}".format(epoch, train_loss, test_loss))
            
    def eval_model(self, data, test=False):
        """Evaluate network"""
            
        x, edge_index = data.x, data.edge_index
            
        with torch.no_grad():
            out = self.forward_test(x, edge_index, self.kernel).cpu()
            
        if test:
            out = out[data.test_mask]

        return out
    

def loss_comp(out):
    """
    Unsupervised loss from Hamilton et al. 2018, using negative sampling.

    """
    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
    loss = (-pos_loss - neg_loss)/out.shape[0]
    
    return loss