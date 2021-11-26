#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path as osp
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import subgraph
from torch_geometric.datasets import Planetoid
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader

from sage_utils import NeighborSampler, SAGE, loss_comp, model_eval
from sage_postproc import model_vis

# data = torch.load('data.pt')

def main():
    
#for more parameters
#optimizer = torch.optim.Adam([
    # dict(params=model.conv1.parameters(), weight_decay=5e-4),
    # dict(params=model.conv2.parameters(), weight_decay=0)
# ], lr=0.01)

    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    
    par = {'hidden_channels': 64,
           'batch_size': 200,
           'num_layers': 3,
           'epochs': 2,
           'lr': 0.01}
    
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # build model
    model = SAGE(data.num_node_features, 
                 hidden_channels=par['hidden_channels'], 
                 num_layers=par['num_layers'])
    
    model = train(model, data, par, writer)
         
    emb, _, _ = model_eval(model, data) 
    model_vis(emb)
    

def train(model, data, par, writer):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    loader = NeighborSampler(data.edge_index, 
                             sizes=[10, 10], 
                             batch_size=par['batch_size'],
                             shuffle=True, 
                             num_nodes=data.num_nodes)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=par['lr'])

    x, edge_index = data.x.to(device), data.edge_index.to(device)
    for epoch in range(1, par['epochs']):
        total_loss = 0
        model.train()
        
        for _, n_id, adjs in loader:
            optimizer.zero_grad() #zero gradients, otherwise accumulates gradients
            
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]

            # compute the model for only the nodes in batch n_id
            out = model(x[n_id], adjs)
            
            loss = loss_comp(out)
        
            #backprop
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * out.size(0)    
        
        total_loss /= data.num_nodes       
        writer.add_scalar("loss", total_loss, epoch)
        print("Epoch {}. Loss: {:.4f}. ".format(
                epoch, total_loss))

    return model
    
          
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit