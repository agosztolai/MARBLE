#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP

from tensorboardX import SummaryWriter
from datetime import datetime

from GeoDySys import geometry, utils, layers, dataloader

"""Main network"""
class net(nn.Module):
    def __init__(self, 
                 data,
                 local_gauge=False, 
                 include_identity=False,
                 **kwargs):
        super(net, self).__init__()
        
        self.par = utils.parse_parameters(self, data, kwargs)
        self.include_identity = include_identity
        
        #gauges
        gauges, R = geometry.compute_gauges(data, local_gauge, self.par['n_geodesic_nb'])
        
        #kernels
        # self.kernel = geometry.gradient_op(data)
        self.kernel = geometry.DD(data, gauges)
            
        #Laplacians
        L = geometry.compute_laplacian(data)
        Lc = geometry.compute_connection_laplacian(data, R)
        
        #diffusion
        nt = self.par['n_scales']
        dim = data.x.shape[1]
        scales = torch.linspace(0,self.par['large_scale'], nt)
        self.diffusion = nn.ModuleList() 
        for tau in scales:
            self.diffusion.append(layers.Diffusion(L, Lc, ic=tau))
            
        #conv layers
        self.convs = nn.ModuleList()
        in_channels = dim*nt
        cum_channels = 0
        for i in range(self.par['order'] + self.par['depth']):
            # in_channels *= len(self.kernel)
            if include_identity:
                in_channels += 1
            if i < self.par['order']:
                conv = layers.AnisoConv(in_channels, 
                                        convert_to_energy=True,
                                        )
            else:
                conv = layers.AnisoConv(in_channels, 
                                        vanilla_GCN=True,
                                        lin_trnsf=True,
                                        ReLU=True,
                                        vec_norm=self.par['vec_norm'],
                                        )
            self.convs.append(conv)
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
        
        utils.print_settings(self, in_channels)
        
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
    def forward(self, x, n_id=None, adjs=None):
        """Forward pass. 
        Messages are passed to a set target nodes (current batch) from sources 
        nodes. The source nodes are sampled randomly as one-step random walks 
        starting from target nodes. Thus the source nodes and target nodes form 
        a bipartite graph to simplify message passing. By convention, the target n
        odes are placed first in variable x, i.e, x = concat[x_target, x_other]."""
        
        #diffusion
        vector = True if x.shape[-1] > 1 else False
        x = [d(x, vector) for  d in self.diffusion]
        x = torch.cat(x, axis=1)
        
        #restrict to current batch
        K = self.kernel
        if n_id is not None:
            x = x[n_id] #n_id are the node ids in the batch
            if K is not None:
                K = [K_[n_id,:][:,n_id] for K_ in K]
            
        #convolutions
        out = [x] if self.include_identity else []
        for i, (edge_index, _, size) in enumerate(adjs):
            #by convention, the first size[1] nodes are the targets
            x = self.convs[i]((x, x[:size[1]]), edge_index, K=K)
            out.append(x)
            
        out = [o[:size[1]] for o in out] #only take target nodes
        out = torch.cat(out, axis=1)
        
        return self.MLP(out)
    
    def evaluate(self, data):
        """Forward pass @ evaluation (no minibatches)"""            
        with torch.no_grad():
            size = (data.x.shape[0], data.x.shape[0])
            adjs = [[data.edge_index, None, size]]
            adjs *= (self.par['order'] + self.par['depth'])
            
            return self(data.x, None, adjs)
    
    def batch_evaluate(self, x, adjs, n_id, device):
        """
        Evaluate network in batches. Launch forward()

        Parameters
        ----------
        x : (nxdim) feature matrix
        adjs : holds a list of `(edge_index, e_id, size)` tuples.
        n_id : list of node ids for current batch
        device : device, default is 'cpu'

        """
        if not isinstance(adjs, list):
            adjs = [adjs]
        adjs = [adj.to(device) for adj in adjs]
        
        return self(x, n_id, adjs)
    
    def train_model(self, data):
        """Network training"""
        
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader, val_loader, test_loader = dataloader.loaders(data, self.par)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.par['lr'])
        self = self.to(device)
        x = data.x.to(device)
        
        print('\n---- Starting training ... \n')
        
        for epoch in range(self.par['epochs']): #loop over epochs
            
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
                  .format(epoch+1, train_loss, val_loss))
        
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