#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from .dataloader import NeighborSampler
from sklearn.model_selection import train_test_split


def train(model, data, par, writer):
    """
    Network training function.

    Parameters
    ----------
    model : pytorch network module
    data : pytorch geometric data object containing
            .edge_index, .num_nodes, .x, .kernels (optional)
           If .kernels is omitted, then adjacency matrix 
           is used for isotropic convolutions (vanilla GCN)
    par : dict
        Parameter values for training.
    writer : tensorboardX summary writer object

    Returns
    -------
    model : pytorch network module
        Trained network.

    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if np.isscalar(par['n_neighbours']):
        par['n_neighbours'] = [par['n_neighbours'] for i in range(par['n_conv_layers'])]
    
    loader = NeighborSampler(data.edge_index,
                             sizes=par['n_neighbours'],
                             batch_size=par['batch_size'],
                             shuffle=True, 
                             num_nodes=data.num_nodes,
                             dropout=par['edge_dropout'],
                             )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=par['lr'])

    #loop over epochs
    x = data.x.to(device)
    for epoch in range(1, par['epochs']):
        total_loss = 0
        model.train()
        
        for _, n_id, adjs in loader:
            optimizer.zero_grad() #zero gradients, otherwise accumulates gradients
            
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            if not isinstance(adjs, list):
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]
                
            #take submatrix corresponding to current batch
            if hasattr(data, 'kernels'):
                if isinstance(data.kernels, list):
                    K = [K_[n_id,:][:,n_id] for K_ in data.kernels]
                else:
                    K = [data.kernels[n_id,:][:,n_id]]
            else:
                K = None
            
            out = model(x[n_id], adjs, K)
            loss = loss_comp(out)
            loss.backward() #backprop
            optimizer.step()

            total_loss += float(loss) * out.size(0)    
        
        total_loss /= data.num_nodes       
        writer.add_scalar("loss", total_loss, epoch)
        print("Epoch {}. Loss: {:.4f}. ".format(
                epoch, total_loss))

    return model


def model_eval(model, data):
    """
    Network evaluating function.

    Parameters
    ----------
    model : pytorch network module
    data : pytorch geometric data object containing
            .edge_index, .num_nodes, .x, .kernels (optional)
           If .kernels is omitted, then adjacency matrix 
           is used for isotropic convolutions (vanilla GCN)

    Returns
    -------
    out : torch tensor
        network output.

    """
    model.eval()
    
    if hasattr(data, 'kernels'):
        if isinstance(data.kernels, list):
            K = [K_ for K_ in data.kernels]
        else:
            K = [data.kernels]
    else:
        K = None
        
    x, edge_index = data.x, data.edge_index
    with torch.no_grad():
        out = model.full_forward(x, edge_index, K).cpu()

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


def split(data, test_size=0.1, val_size=0.5, seed=0):
    """
    

    Parameters
    ----------
    data : pytorch geometric data object. All entried must be torch tensors
            for this to work properly.
    test_size : float between 0 and 1, optional
        Test set as fraction of total dataset. The default is 0.1.
    val_size : float between 0 and 1, optional
        Validation set as fraction of test set. The default is 0.5.
    seed : int, optional
        Seed. The default is 0.

    Returns
    -------
    data : pytorch geometric data object.

    """
    
    n = len(data.x)
    train_id, test_id = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    test_id, val_id = train_test_split(test_id, test_size=val_size, random_state=seed)
    
    train_mask = torch.zeros(n, dtype=bool)
    test_mask = torch.zeros(n, dtype=bool)
    val_mask = torch.zeros(n, dtype=bool)
    train_mask[train_id] = True
    test_mask[test_id] = True
    val_mask[val_id] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    
    return data