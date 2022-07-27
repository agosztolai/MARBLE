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
                 **kwargs):
        super(net, self).__init__()
        
        self = utils.parse_parameters(self, data, kwargs)
        
        self.vector = True if self.dim > 1 else False
        
        #gauges
        gauges, self.R = geometry.compute_gauges(data, local_gauge, self.par['n_geodesic_nb'])
        
        #kernels
        # self.kernel = geometry.gradient_op(data)
        self.kernel = geometry.DD(data, gauges)
        
        #diffusion
        self.diffusion = layers.Diffusion(data)
            
        #gradient features
        cum_channels = 0
        self.grad = nn.ModuleList()
        for i in range(self.par['order']):
            self.grad.append(layers.AnisoConv())
            cum_channels += self.dim
            cum_channels *= len(self.kernel)
            
        cum_channels += self.dim
            
        #message passing
        self.convs = nn.ModuleList()
        for i in range(self.par['depth']):
            conv = layers.AnisoConv(cum_channels, 
                                    vanilla_GCN=True,
                                    lin_trnsf=True,
                                    ReLU=True,
                                    vec_norm=self.par['vec_norm'],
                                    )
            self.convs.append(conv)
            
        #sheaf learning
        self.sheaf = layers.SheafLearning(self.dim, data.x)
            
        #inner product features
        self.inner_products = layers.InnerProductFeatures(self.dim, self.dim)
            
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
        
        utils.print_settings(self, cum_channels)
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
    def forward(self, x, n_id=None, adjs=None):
        """Forward pass. 
        Messages are passed to a set target nodes (current batch) from source
        nodes. The source nodes are sampled randomly by one-step random walks 
        starting from target nodes. Thus the source nodes and target nodes form 
        a bipartite graph to simplify message passing. By convention, the target n
        odes are placed first in variable x, i.e, x = concat[x_target, x_other]."""
        
        #diffusion
        x = self.diffusion(x, self.vector)
        
        #restrict to current batch
        K = self.kernel
        if n_id is not None:
            x = x[n_id] #n_id are the node ids in the batch
            if K is not None:
                K = [K_[n_id,:][:,n_id] for K_ in K]

        #gradients
        out = [x]
        adjs_ = adjs[:self.par['order']]
        for i, (edge_index, _, size) in enumerate(adjs_):
            R = self.sheaf(out[0], edge_index)
                # Lc = geometry.compute_connection_laplacian(data, R)
            
            #by convention, the first size[1] nodes are the targets
            x = self.grad[i]((x, x[:size[1]]), edge_index, K=K)          
            # x = self.inner_products(x)
            out.append(x)
            
        out = [o[:size[1]] for o in out] #only take target nodes
        out = torch.cat(out, axis=1)
            
        #message passing
        adjs_ = adjs[self.par['order']:]
        for i, (edge_index, _, size) in enumerate(adjs_):
            out = self.convs[i]((out, out[:size[1]]), edge_index)          
        
        return self.MLP(out), R
    
    def evaluate(self, data):
        """Forward pass @ evaluation (no minibatches)"""            
        with torch.no_grad():
            size = (data.x.shape[0], data.x.shape[0])
            adjs = [[data.edge_index, None, size]]
            adjs *= (self.par['order'] + self.par['depth'])
            
            return self(data.x, None, adjs)
    
    def evaluate_batch(self, x, batch, device):
        """
        Evaluate network in batches.

        Parameters
        ----------
        x : (nxdim) feature matrix
        batch : triple containing
            n_id : list of node ids for current batch
            adjs : list of `(edge_index, e_id, size)` tuples.
        device : device, default is 'cpu'

        """
        
        _, n_id, adjs = batch
        
        if not isinstance(adjs, list):
            adjs = [adjs]
        adjs = [adj.to(device) for adj in adjs]
        
        out, R = self.forward(x, n_id, adjs)

        return compute_loss(out, x, batch, R)
    
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
            for batch in train_loader: #loop over batches
                optimizer.zero_grad() #zero gradients, otherwise accumulates gradients
                loss = self.evaluate_batch(x, batch, device)
                train_loss += float(loss)
                
                loss.backward() #backprop
                optimizer.step()
                                
            self.eval() #switch to testing mode (this disables dropout in MLP)
            val_loss = 0
            for batch in val_loader: #loop over batches    
                loss = self.evaluate_batch(x, batch, device)
                val_loss += float(loss)  
            val_loss /= (sum(data.val_mask)/sum(data.train_mask))
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            print("Epoch: {},  Training loss: {:.4f}, Validation loss: {:.4f}"\
                  .format(epoch+1, train_loss, val_loss))
        
        test_loss = 0
        for _, n_id, adjs in test_loader: #loop over batches                
            loss = self.evaluate_batch(x, batch, device)
            test_loss += float(loss)
        test_loss /= (sum(data.test_mask)/sum(data.train_mask))
        print('Final test loss: {:.4f}'.format(test_loss))
    

def compute_loss(out, x, batch, R):
    """Unsupervised loss modified from from GraphSAGE (Hamilton et al. 2018.)"""
    
    z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
    pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()
    
    if x.shape[1] == 1:
        return - pos_loss - neg_loss
    
    _, n_id, adjs = batch
    edge_index = adjs[-1].edge_index
    
    Rij = R[edge_index[0], edge_index[1], ...]
    xi = x[n_id][edge_index[1]]
    xj = x[n_id][edge_index[0]]
    
    #compute sum_ij || R_ij * x(j) - x(i) || via broadcasting
    R_loss = torch.einsum('aij,aj->ai',Rij, xj) - xi
    R_loss = F.logsigmoid((R_loss).norm(dim=1)).mean()
    
    #orthogonality constraint sum_ij || Rij*Rij^T - I ||
    RRT = torch.matmul(Rij, Rij.swapaxes(1,2)) #batch matrix multiplication
    eye = torch.eye(2).unsqueeze(0).repeat(len(edge_index.T),1,1)
    orth_loss = ((RRT-eye).norm(dim=(1,2))).mean()
    
    return - pos_loss - neg_loss + orth_loss