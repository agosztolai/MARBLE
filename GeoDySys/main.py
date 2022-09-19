#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from datetime import datetime

from .lib import utils, geometry
from . import preprocessing, layers, dataloader


"""Main network"""
class net(nn.Module):
    def __init__(self, 
                 data,
                 **kwargs):
        super(net, self).__init__()
        
        #parameters
        self.par = utils.parse_parameters(data, kwargs)
        
        #preprocessing
        self.gauges, self.R, self.kernel = preprocessing(data, self.par)
        
        #layers
        self.diffusion, self.grad, self.convs, self.mlp, self.inner_products = \
            layers.setup_layers(data, self.R, self.par)
            
        self.reset_parameters()
        
        utils.print_settings(self)
        
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
        x = self.diffusion(x)
        
        #restrict to current batch
        kernels = self.kernel
        R = self.R
        if n_id is not None: #n_id are the node ids in the batch
            x = x[n_id] 
            kernels = [K[n_id,:][:,n_id] for K in kernels]
            R = self.R[n_id,:][:,n_id] if R is not None else None

        #gradients
        out = []
        adjs_ = adjs[:self.par['order']]
        for i, (edge_index, _, size) in enumerate(adjs_):            
                        
            x = self.grad[i](x, edge_index, size, kernels, R)          
            # x = self.inner_products(x)
            out.append(x)
            
        out = [o[:size[1]] for o in out] #only take target nodes
        out = torch.cat(out, axis=1)
            
        #message passing
        adjs_ = adjs[self.par['order']:]
        for i, (edge_index, _, size) in enumerate(adjs_):
            out = self.convs[i]((out, out[:size[1]]), edge_index)          
        
        return self.mlp(out)
    
    def evaluate(self, data):
        """Forward pass @ evaluation (no minibatches)"""            
        with torch.no_grad():
            size = (data.x.shape[0], data.x.shape[0])
            adjs = [[data.edge_index, None, size]]
            adjs *= (self.par['order'] + self.par['depth'])
            
            self.emb = self(data.x, None, adjs)
                
    def evaluate_batch(self, x, batch, device='cpu'):
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
        
        adjs = utils.to_list(adjs)
        adjs = [adj.to(device) for adj in list(adjs)]
        
        out = self.forward(x, n_id, adjs)

        return compute_loss(out, x)
    
    def run_training(self, data):
        """Network training"""
        
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader, val_loader, test_loader = dataloader.loaders(data, self.par)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.par['lr'])
        self = self.to(device)
        x = data.x.to(device)
        
        print('\n---- Starting training ... \n')
        
        for epoch in range(self.par['epochs']):
            
            self.train() #training mode
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad() #zero gradients, otherwise accumulates
                loss = self.evaluate_batch(x, batch, device)
                train_loss += float(loss)
                
                loss.backward() #backprop
                optimizer.step()
                                
            self.eval() #testing mode (disables dropout in MLP)
            val_loss = 0
            for batch in val_loader:  
                loss = self.evaluate_batch(x, batch, device)
                val_loss += float(loss)  
            val_loss /= (sum(data.val_mask)/sum(data.train_mask))
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            print("Epoch: {}, Training loss: {:.4f}, Validation loss: {:.4f}" \
                  .format(epoch+1, train_loss, val_loss))
        
        test_loss = 0
        for batch in test_loader:               
            loss = self.evaluate_batch(x, batch, device)
            test_loss += float(loss)
        test_loss /= (sum(data.test_mask)/sum(data.train_mask))
        print('Final test loss: {:.4f}'.format(test_loss))
        
        
    def cluster_and_embed(self,
                          cluster_typ='kmeans', 
                          embed_typ='tsne', 
                          n_clusters=15, 
                          seed=0):
        """
        Cluster & embed
        
        Returns
        -------
        emb : nx2 matrix of embedded data
        clusters : sklearn cluster object
        dist : cxc matrix of pairwise distances where c is the number of clusters
        
        """

        clusters = geometry.cluster(self.emb, cluster_typ, n_clusters, seed)
        clusters = geometry.relabel_by_proximity(clusters)
        clusters['slices'] = self.par['slices']
        
        emb = np.vstack([self.emb, clusters['centroids']])
        emb = geometry.embed(emb, embed_typ)  
        emb, clusters['centroids'] = emb[:-n_clusters], emb[-n_clusters:]
        
        dist = geometry.compute_distr_distances(clusters)
            
        return emb, clusters, dist
    

def compute_loss(out, x):
    """Unsupervised loss modified from from GraphSAGE (Hamilton et al. 2018.)"""
    
    z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
    pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()
    
    return - pos_loss - neg_loss