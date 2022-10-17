#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from GeoDySys import plotting, utils, geometry, net

def main():
    
    #parameters
    n = 512
    k = 20
    n_clusters = 10
    
    par = {'batch_size': 256, #batch size, this should be as large as possible
           'epochs': 3, #optimisation epochs
           'order': 2, #order of derivatives
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
    model = net(data, gauge='global', **par)
    model.run_training(data)
    model.evaluate(data)
    emb, clusters, dist = model.cluster_and_embed(n_clusters=n_clusters)
    
    #plot
    titles=['Constant','Linear','Parabola','Saddle']
    plotting.fields(data, titles=titles, save='scalar_fields.svg')
    plotting.embedding(emb, data.y.numpy(), clusters, titles=titles, save='scalar_fields_embedding.svg') 
    plotting.histograms(clusters, titles=titles, save='scalar_fields_histogram.svg')
    plotting.neighbourhoods(data, clusters, hops=1, norm=True, save='scalar_fields_nhoods.svg') 
    

# def f0(x, alpha=1):
#     return np.cos(alpha)*x[:,[0]] + np.sin(alpha)*x[:,[1]]
  
# def f1(x, alpha=2):
#     return np.cos(alpha)*x[:,[0]] + np.sin(alpha)*x[:,[1]]

# def f2(x, alpha=3):
#     return np.cos(alpha)*x[:,[0]] + np.sin(alpha)*x[:,[1]]

# def f3(x, alpha=4):
#     return np.cos(alpha)*x[:,[0]] + np.sin(alpha)*x[:,[1]]
    
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