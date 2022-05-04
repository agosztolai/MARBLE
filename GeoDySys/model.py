#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from datetime import datetime
from .dataloader import NeighborSampler
from .layers import AnisoConv, MLP
from .kernels import aggr_directional_derivative

import numpy as np

"""Main network architecture"""
class net(nn.Module):
    def __init__(self, data, kernel='directional_derivative', gauge='global', **kwargs):
        super(net, self).__init__()
        
        self.n_layers = kwargs['n_layers'] if 'n_layers' in kwargs else 1
        
        if kernel == 'directional_derivative':
            self.kernel = aggr_directional_derivative(data, gauge)
        else:
            self.kernel = None
        
        #initialize conv layers
        self.convs = nn.ModuleList() #could use nn.Sequential because we execute in order
        for i in range(self.n_layers):
            self.convs.append(AnisoConv(
                adj_norm=kwargs['adj_norm'] if 'adj_norm' in kwargs else False))
        
        #initialize multilayer perceptron
        self.MLP = MLP(in_channels=data.x.shape[1] if self.kernel is None else len(self.kernel)*data.x.shape[1],
                       hidden_channels=kwargs['hidden_channels'], 
                       out_channels=kwargs['out_channels'] if 'out_channels' in kwargs else None,
                       n_lin_layers=kwargs['n_lin_layers'] if 'n_lin_layers' in kwargs else 1,
                       activation=kwargs['activation'] if 'activation' in kwargs else False,
                       dropout=kwargs['dropout'] if 'dropout' in kwargs else 0.,
                       b_norm=kwargs['b_norm'] if 'b_norm' in kwargs else False)
        
    def forward(self, x, adjs, K=None):
        """forward pass"""
        #loop over minibatches
        for i, (edge_index, _, size) in enumerate(adjs):
            #messages are passed from x_source (all nodes) to x_target
            x_source = x 
            x_target = x[:size[1]]
            
            #convolution
            x = self.convs[i]((x_source, x_target), edge_index, K=K, edge_weight=None, size=size)
            
            #multilayer perceptron
            x = self.MLP(x)
                                              
        return x

    # for testing, we don't do minibatch
    def full_forward(self, x, edge_index, K=None):        
        for conv in self.convs:
            x = conv(x, edge_index, K=K, size=(x.shape[0],x.shape[0]))
            x = self.MLP(x)
            
        return x        
    
    
    def train_model(self, data, par):
        
        """
        Network training function.

        Parameters
        ----------
        data : pytorch geometric data object containing
                .edge_index, .num_nodes, .x, .kernels (optional)
               If .kernels is omitted, then adjacency matrix 
               is used for isotropic convolutions (vanilla GCN)
        par : dict
            Parameter values for training.

        """
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self = self.to(device)
        
        if np.isscalar(par['n_neighbours']):
            par['n_neighbours'] = [par['n_neighbours'] for i in range(par['n_conv_layers'])]
        
        loader = NeighborSampler(data.edge_index,
                                 sizes=par['n_neighbours'],
                                 batch_size=par['batch_size'],
                                 shuffle=True, 
                                 num_nodes=data.num_nodes,
                                 dropout=par['edge_dropout'],
                                 )
        
        optimizer = torch.optim.Adam(self.parameters(), lr=par['lr'])

        #loop over epochs
        x = data.x.to(device)
        for epoch in range(1, par['epochs']):
            total_loss = 0
            self.train()
            
            for _, n_id, adjs in loader:
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

                total_loss += float(loss) * out.size(0)    
            
            total_loss /= data.num_nodes       
            writer.add_scalar("loss", total_loss, epoch)
            print("Epoch {}. Loss: {:.4f}. ".format(
                    epoch, total_loss))
            
            
    def eval_model(self, data):
        """
        Network evaluating function.

        Parameters
        ----------
        data : pytorch geometric data object containing
                .edge_index, .num_nodes, .x, .kernels (optional)
               If .kernels is omitted, then adjacency matrix 
               is used for isotropic convolutions (vanilla GCN)

        Returns
        -------
        out : torch tensor
            network output.

        """
        self.eval()
        
        if self.kernel is not None:
            K = [K_ for K_ in self.kernel]
        else:
            K = None
            
        x, edge_index = data.x, data.edge_index
        with torch.no_grad():
            out = self.full_forward(x, edge_index, K).cpu()

        return out
    

def loss_comp(out):
    """
    Unsupervised loss function from Hamilton et al. 2018, using negative sampling.

    Parameters
    ----------
    out : pytorch tensor
        Output of network.
    Returns
    -------
    loss : float
        Loss.

    """
    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

    #loss function from word2vec
    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
    loss = (-pos_loss - neg_loss)/out.shape[0]
    
    return loss