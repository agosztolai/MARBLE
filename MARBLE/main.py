#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from datetime import datetime

from .lib import utils, geometry
from . import layers, dataloader

from tqdm import tqdm

"""Main network"""
class net(nn.Module):
    def __init__(self, data, loadpath=None, **kwargs):
        super(net, self).__init__()
        
        self.epoch = 0
        self.par = utils.parse_parameters(data, kwargs)
        self = layers.setup_layers(self)      
        self.loss = loss_fun()       
        self.reset_parameters()
        
        utils.print_settings(self)
        
        if loadpath is not None:
            self.load_model(loadpath)
        
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        
    def forward(self, data, n_id, adjs=None):
        """Forward pass. 
        Messages are passed to a set target nodes (current batch) from source
        nodes. The source nodes and target nodes form a bipartite graph to 
        simplify message passing. By convention, the first size[1] entries of x 
        are the target nodes, i.e, x = concat[x_target, x_other]."""
        
        x = data.x
        n, d = data.x.shape[0], data.gauges.shape[2]
        
        #local gauges
        if self.par['inner_product_features']:
            x = geometry.map_to_local_gauges(x, data.gauges)   
                
        #diffusion
        if self.par['diffusion']:            
            L = data.L.copy() if hasattr(data, 'L') else None
            Lc = data.Lc.copy() if hasattr(data, 'Lc') else None
            x = self.diffusion(x, L, Lc=Lc, method='spectral')
            
        #restrict to current batch
        x = x[n_id]
        if data.kernels[0].size(0) == n*d:
            n_id = utils.expand_index(n_id, d)
        else:
            d=1
    
        if self.par['vec_norm']:
            x = F.normalize(x, dim=-1, p=2)
        
        #gradients
        out = [x]
        for i, (edge_index, _, size) in enumerate(adjs):
            kernels = [K[n_id[:size[1]*d], :][:, n_id[:size[0]*d]] for K in data.kernels]
            
            x = self.grad[i](x, kernels)
            out.append(x)
            
        out = [o[:size[1]] for o in out] #take target nodes
            
        #inner products
        if self.par['inner_product_features']:
            out = self.inner_products(out)
        else:
            out = torch.cat(out, axis=1)     
            
        if self.par['include_positions']:
            out = torch.hstack([data.pos[n_id[:size[1]]], out])
        
        emb = self.enc(out)
        
        return emb
    
    
    def evaluate(self, data):
        """Forward pass @ evaluation (no minibatches)"""            
        with torch.no_grad():
            size = (data.x.shape[0], data.x.shape[0])
            adjs = utils.EdgeIndex(data.edge_index, 
                                   torch.arange(data.edge_index.shape[1]), 
                                   size)
            adjs = utils.to_list(adjs) * self.par['order']
            
            try:
                data.kernels = [utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values()).t() for K in utils.to_list(data.kernels)]
            except:
                pass
        
            #load to gpu if possible
            model, data.x, data.pos, data.L, data.Lc, data.kernels, data.gauges, adjs = \
                utils.move_to_gpu(self, data, adjs)
                
            out = self.forward(data, torch.arange(len(data.x)), adjs)
            
            model, data.x, data.pos, data.L, data.Lc, data.kernels, data.gauges, adjs = \
                utils.detach_from_gpu(self, data, adjs)
            
            data.out = out.detach().cpu()
            
            return data
                

    def batch_loss(self, data, loader, train=False, verbose=False, optimizer=None):
        """Loop over minibatches provided by loader function.
        
        Parameters
        ----------
        x : (nxdim) feature matrix
        loader : dataloader object from dataloader.py
        
        """
        
        if train: #training mode (enables dropout in MLP)
            self.train()
        
        if verbose:
            print('\n')
            
        cum_loss = 0
        for batch in tqdm(loader, disable=not verbose):
            _, n_id, adjs = batch
            adjs = [adj.to(data.x.device) for adj in utils.to_list(adjs)]
                        
            emb = self.forward(data, n_id, adjs)
            loss = self.loss(emb)
            cum_loss += float(loss)
            
            if optimizer is not None:
                optimizer.zero_grad() #zero gradients, otherwise accumulates
                loss.backward() #backprop
                optimizer.step()
                
        self.eval()
                
        return cum_loss/len(loader), optimizer
    
    
    def run_training(self, data, outdir=None, use_best=True, verbose=False):
        """Network training"""
        
        print('\n---- Training network ...')
        
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
                
        #load to gpu (if possible)
        self, data.x, data.pos, data.L, data.Lc, data.kernels, data.gauges = utils.move_to_gpu(self, data)
        
        #data loader
        train_loader, val_loader, test_loader = dataloader.loaders(data, self.par)
        optimizer = opt.SGD(self.parameters(), lr=self.par['lr'], momentum=self.par['momentum'])
        if hasattr(self, 'optimizer_state_dict'):
            optimizer.load_state_dict(self.optimizer_state_dict)
        writer = SummaryWriter('./log/')
        
        #training scheduler
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        best_loss = -1
        for epoch in range(self.par['epochs']):
            
            epoch = self.epoch + epoch + 1
            
            train_loss, optimizer = self.batch_loss(data, train_loader, train=True, verbose=verbose, optimizer=optimizer)
            val_loss, _ = self.batch_loss(data, val_loader, verbose=verbose)
            scheduler.step(train_loss)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.flush()
            print("\nEpoch: {}, Training loss: {:.4f}, Validation loss: {:.4f}, lr: {:.4f}" \
                  .format(epoch, train_loss, val_loss, scheduler._last_lr[0]), end="")
                
            if best_loss==-1 or (val_loss<best_loss):
                outdir = self.save_model(optimizer, outdir, best=True, timestamp=time)
                best_loss = val_loss 
                print(' *', end="")
        
        test_loss, _ = self.batch_loss(data, test_loader)
        writer.add_scalar('Loss/test', test_loss)
        writer.close()
        print('\nFinal test loss: {:.4f}'.format(test_loss))
        
        self.save_model(optimizer, outdir, best=False, timestamp=time)
        
        if use_best:
            self.load_model(os.path.join(outdir, 'best_model_{}.pth'.format(time)))
            
            
    def load_model(self, loadpath):
        
        checkpoint = torch.load(loadpath)
        self.epoch = checkpoint['epoch']
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_state_dict = checkpoint['optimizer_state_dict']
                                

    def save_model(self, optimizer, outdir=None, best=False, timestamp=""):
        
        if outdir is None:
            outdir = './outputs/'   
             
        if not os.path.exists(outdir):
            os.makedirs(outdir)
                
        checkpoint = {'epoch': self.epoch,
                      'model_state_dict': self.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'time': timestamp
                     }
        
        if best:
            fname = 'best_model_'
        else:
            fname = 'last_model_'
            
        fname += timestamp
        fname += '.pth'
                                
        if best:
            torch.save(checkpoint, os.path.join(outdir,fname))
        else:
            torch.save(checkpoint, os.path.join(outdir,fname))
            
        return outdir


class loss_fun(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out):
            
        z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()
        
        return -pos_loss -neg_loss