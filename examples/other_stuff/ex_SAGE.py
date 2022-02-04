#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path as osp
import sys
from datetime import datetime

# import torch
# import torch.nn.functional as F

import torch_geometric.transforms as T
# import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils
# from torch_geometric.utils import subgraph
from torch_geometric.datasets import Planetoid
from tensorboardX import SummaryWriter
# from torch_geometric.loader import DataLoader

from GeoDySys.embedding import SAGE, model_eval, train


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
    
    from torch_geometric.data import InMemoryDataset

    InMemoryDataset.collate([data,data])
    
    par = {'hidden_channels': 64,
           'batch_size': 200,
           'num_layers': 3,
           'epochs': 20,
           'lr': 0.01}
    
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # build model
    model = SAGE(data.num_node_features, 
                 hidden_channels=par['hidden_channels'], 
                 num_layers=par['num_layers'])
    
    model = train(model, data, par, writer)
         
    emb = model_eval(model, data) 
    model_vis(emb, data)


def model_vis(emb, data):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    colors = ["red", "orange", "green", "blue", "purple", "brown", "black"]
    colors = [colors[y] for y in data.y]
    xs, ys = zip(*TSNE().fit_transform(emb.detach().numpy()))
    plt.scatter(xs, ys, color=colors)

    
          
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit