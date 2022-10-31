#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from MARBLE import plotting, utils, geometry, net

def main():
    
    #parameters
    n = 512
    k = 20
    n_clusters = 10
    
    par = {'batch_size': 128, #batch size
           'epochs': 30, #optimisation epochs
           'order': 1, #order of derivatives
           'n_lin_layers': 2,
           'hidden_channels': 16, #number of internal dimensions in MLP
           'out_channels': 4,
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
    emb, clusters, dist = geometry.cluster_embedding(data, n_clusters=n_clusters)
    
    #plot
    titles=['Constant','Linear','Parabola','Saddle']
    plotting.fields(data, titles=titles, node_size=30, save='scalar_fields.svg')
    plotting.embedding(emb, data.y.numpy(), clusters, titles=titles, save='scalar_fields_embedding.svg') 
    plotting.histograms(clusters, titles=titles, save='scalar_fields_histogram.svg')
    plotting.neighbourhoods(data, clusters, hops=1, norm=True, save='scalar_fields_nhoods.svg') 
    
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