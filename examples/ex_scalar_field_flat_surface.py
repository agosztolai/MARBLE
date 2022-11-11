#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from MARBLE import plotting, utils, geometry, net

def main():
    
    #parameters
    n = 512
    k = 30
    n_clusters = 16
    
    par = {'epochs': 30, #optimisation epochs
           'order': 1, #order of derivatives
           'n_lin_layers': 1,
           'hidden_channels': 8, #number of internal dimensions in MLP
           'out_channels': 3,
           'inner_product_features': False,
           }
    
    #evaluate functions
    # f1: constant, f2: linear, f3: parabola, f4: saddle
    x = [geometry.sample_2d(n, [[-1,-1],[1,1]], 'random') for i in range(4)]
    y = [f0(x[0]), f1(x[1]), f2(x[2]), f3(x[3])] #evaluated functions
        
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
    
    #train model
    model = net(data, **par)
    model.run_training(data)
    
    #evaluate model on data
    data = model.evaluate(data)
    emb, _, clusters, dist, _ = geometry.cluster_embedding(data, n_clusters=n_clusters)
    
    #plot
    titles=['Constant','Linear','Parabola','Saddle']
    plotting.fields(data, titles=titles, node_size=10, col=2)
    plotting.embedding(emb, data.y.numpy(), clusters, titles=titles)
    plotting.histograms(clusters, titles=titles)
    plotting.neighbourhoods(data, clusters, hops=1, norm=True,figsize=(10, 20))
    
def f0(x):
    return x[:,[0]]*0

def f1(x):
    return x[:,[0]] + x[:,[1]]

def f2(x):
    return x[:,[0]]**2 + x[:,[1]]**2

def f3(x):
    return x[:,[0]]**2 - x[:,[1]]**2


if __name__ == '__main__':
    sys.exit(main())