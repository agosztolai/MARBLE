#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from datetime import datetime

from .lib import utils
from . import layers, dataloader


"""Main network"""
class net(nn.Module):
    def __init__(self, data, **kwargs):
        super(net, self).__init__()
        
        assert all(hasattr(data, attr) for attr in ['kernels', 'L']), \
            'It seems that data is not preprocessed. Run preprocess(data) \
            before initialising network!'
        
        self.par = utils.parse_parameters(data, kwargs)
        self = layers.setup_layers(self)      
        self.loss = loss_fun()       
        self.reset_parameters()
        
        utils.print_settings(self)
        
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        
    def forward(self, data, n_id=None, adjs=None):
        """Forward pass. 
        Messages are passed to a set target nodes (current batch) from source
        nodes. The source nodes and target nodes form a bipartite graph to 
        simplify message passing. By convention, the first size[1] entries of x 
        are the target nodes, i.e, x = concat[x_target, x_other]."""
        
        x = data.x
        
        #parse parameters
        if n_id is None:
            n_id = np.arange(len(x))

        #diffusion
        if self.par['diffusion']:
            Lc = data.Lc if hasattr(data, 'Lc') else None
            x = self.diffusion(x, data.L, Lc=Lc, method='spectral')
        
        #restrict to current batch n_id
        x = x[n_id] 
        kernels = [K[n_id,:][:,n_id] for K in data.kernels]
        R = data.R[n_id,...][:,n_id,...] if hasattr(data, 'R') else None

        #gradients
        if self.par['vec_norm']:
            out = [F.normalize(x, dim=-1, p=2)]
        else:
            out = [x]
        for i, (edge_index, _, size) in enumerate(adjs):
            x = self.grad[i](x, edge_index, size, kernels, R)
            out.append(x)
            
        #take target nodes
        out = [o[:size[1]] for o in out]
            
        #inner products
        if self.par['inner_product_features']:
            out = self.inner_products(out)
        else:
            out = torch.cat(out, axis=1)     
                    
        emb = self.enc(out)
        if self.par['autoencoder']:
            return emb, out, self.dec(emb)
        else:
            return emb, None, None
    
    
    def evaluate(self, data):
        """Forward pass @ evaluation (no minibatches)"""            
        with torch.no_grad():
            size = (data.x.shape[0], data.x.shape[0])
            adjs = utils.EdgeIndex(data.edge_index, None, size)
            adjs = utils.to_list(adjs) * self.par['order']
            
            #load to gpu if possible
            Lc = data.Lc if hasattr(data, 'Lc') else None
            R = data.R if hasattr(data, 'R') else None
            adjs, data.x, data.L, data.Lc, data.kernels, data.R = \
                utils.move_to_gpu(adjs, data.x, data.L, Lc, data.kernels, R)
            emb, _, _ = self.forward(data, None, adjs)
            data.emb = emb.detach().cpu()
            data.x = data.x.detach().cpu()
            
            return data
                

    def batch_loss(self, data, loader, optimizer=None):
        """Loop over minibatches provided by loader function.
        
        Parameters
        ----------
        x : (nxdim) feature matrix
        loader : dataloader object from dataloader.py
        optimizer : pytorch optimiser
        
        """
        
        cum_loss = 0
        for batch in loader:
            _, n_id, adjs = batch
            adjs = [adj.to(data.x.device) for adj in utils.to_list(adjs)]
                        
            enc_out, out, dec_out = self.forward(data, n_id, adjs)
            loss = self.loss(enc_out, out, dec_out)
            cum_loss += float(loss)
            
            if optimizer is not None:
                optimizer.zero_grad() #zero gradients, otherwise accumulates
                loss.backward() #backprop
                optimizer.step()
                                
        return cum_loss/len(loader), optimizer
    
    
    def run_training(self, data, save=True, loadpath=None):
        """Network training"""
        
        print('\n---- Training network ...')
        
        assert all(hasattr(data, attr) for attr in ['kernels', 'L']), \
            'It seems that data is not preprocessed. Run preprocess(data) before training!'
        
        #load to gpu if possible
        Lc = data.Lc if hasattr(data, 'Lc') else None
        R = data.R if hasattr(data, 'R') else None
        self, data.x, data.L, data.Lc, data.kernels, data.R = \
            utils.move_to_gpu(self, data.x, data.L, Lc, data.kernels, R)
        
        #initialise logger and optimiser
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        train_loader, val_loader, test_loader = dataloader.loaders(data, self.par)
        optimizer = opt.SGD(self.parameters(), lr=self.par['lr'], momentum=self.par['momentum'])
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        if loadpath is not None:
            checkpoint = torch.load(loadpath)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch0 = checkpoint['epoch']
        else:
            checkpoint = {}
            epoch0 = 0
        
        best_loss = -1
        for epoch in range(self.par['epochs']):
            
            self.train() #training mode
            train_loss, optimizer = self.batch_loss(data, train_loader, optimizer)
            
            self.eval() #testing mode (disables dropout in MLP)
            val_loss, _ = self.batch_loss(data, val_loader)
            scheduler.step(train_loss)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            lr = scheduler._last_lr[0]
            print("\nEpoch: {}, Training loss: {:.4f}, Validation loss: {:.4f}, lr: {:.4f}" \
                  .format(epoch+epoch0+1, train_loss, val_loss, lr), end="")
                
            if best_loss==-1 or (val_loss<best_loss):
                best_loss = val_loss 
                checkpoint['model_state_dict'] = self.state_dict()
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                print(' *', end="")
        
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        test_loss, _ = self.batch_loss(data, test_loader)
        print('\nFinal test loss: {:.4f}'.format(test_loss))
        
        if save:
            if not os.path.exists('./outputs'):
                os.makedirs('./outputs')
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, './outputs/best_model.pth')
        

class loss_fun(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lamb = nn.Parameter(torch.tensor(0.5))

    def forward(self, out, *args):
        
        with torch.no_grad():
            self.lamb.data = torch.clamp(self.lamb, min=1e-8, max=1-1e-8)
    
        z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()
        
        if args[0] is None:
            return -pos_loss -neg_loss
        
        x, _, _ = args[0].split(out.size(0) // 3, dim=0)
        xhat, _, _ = args[1].split(out.size(0) // 3, dim=0)
        
        return self.lamb*self.mse(x, xhat) + (1-self.lamb)*(-pos_loss -neg_loss)