#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from GeoDySys import plotting, utils, geometry, net


def main():
    
    #parameters
    n = 512
    k = 30
    n_clusters = 10
    
    par = {'batch_size': 256, #batch size, this should be as large as possible
           'epochs': 2, #optimisation epochs
           'order': 1, #order of derivatives
           'n_lin_layers': 2,
           'hidden_channels': 16, #number of internal dimensions in MLP
           'out_channels': 8,
           }
      
    #evaluate functions
    # f0: linear, f1: point source, f2: point vortex, f3: saddle
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
    titles=['Linear left','Linear right','Vortex right','Vortex left']
    plotting.fields(data, titles=titles, save='vector_fields.svg')
    plotting.embedding(emb, data.y.numpy(), clusters, titles=titles, save='vector_fields_embedding.svg') 
    plotting.histograms(clusters, titles=titles, save='vector_fields_histogram.svg') 
    plotting.neighbourhoods(data, clusters, save='vector_fields_nhoods.svg')
    
def f0(x):
    return x*0 + np.array([-1,-1])

def f1(x):
    return x*0 + np.array([1,1])

# def f2(x):
#     return x*0 + np.array([1,0])

def f3(x):
    return x*0 + np.array([0,1])

# def f1(x):
#     eps = 1e-3
#     norm = np.sqrt(x[:,[0]]**2 + x[:,[1]]**2) + eps
#     u = 2*x[:,[0]]/norm
#     v = 2*x[:,[1]]/norm
#     return np.hstack([u,v])

def f2(x):
    eps = 1e-1
    norm = np.sqrt((x[:,[0]]+1)**2 + x[:,[1]]**2 + eps)
    u = x[:,[1]]/norm
    v = -(x[:,[0]]+1)/norm
    return np.hstack([u,v])

def f3(x):
    eps = 1e-1
    norm = np.sqrt((x[:,[0]]-1)**2 + x[:,[1]]**2 + eps)
    u = x[:,[1]]/norm
    v = -(x[:,[0]]-1)/norm
    return np.hstack([u,v])

# def f3(x):
#     eps = 1e-1
#     norm = x[:,[0]]**2 + x[:,[1]]**2 + eps
#     u = x[:,[1]]/norm
#     v = x[:,[0]]/norm
#     return torch.tensor(np.hstack([u,v])).float() 


if __name__ == '__main__':
    sys.exit(main())