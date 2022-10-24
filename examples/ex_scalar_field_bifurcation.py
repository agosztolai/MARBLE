#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from MARBLE import plotting, utils, geometry, net

def main():
    
    #parameters
    n = 512
    k = 30
    n_clusters = 10
    
    par = {'batch_size': 256, #batch size, this should be as large as possible
           'epochs': 30, #optimisation epochs
           'order': 2, #order of derivatives
           'depth': 0, #number of hops in neighbourhood
           'n_lin_layers': 2,
           'hidden_channels': 16, #number of internal dimensions in MLP
           'out_channels': 4,
           'adj_norm': False,
           }
    
    #evaluate functions
    n_steps = 10
    beta = np.linspace(-1, 1, n_steps)
    x = [geometry.sample_2d(n, [[-1,-1],[1,1]], 'random') for i in range(n_steps)]
    y = [f(x_, beta[i]) for i,x_ in enumerate(x)] #evaluated functions
    
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
    
    #train model
    model = net(data, gauge='global', **par)
    model.run_training(data)
    model.evaluate(data)
    emb, clusters, dist = model.cluster_and_embed(n_clusters=n_clusters)
    emb_MDS = geometry.embed(dist, embed_typ='MDS')
    
    #plot
    titles = [r'$\beta$ = ' + str(a) for a in beta]
    plotting.fields(data, col=5, figsize=(10,3), titles=titles, save='scalar_fields.svg')
    plotting.embedding(emb, data.y.numpy(), clusters, titles=titles, save='scalar_fields_embedding.svg') 
    # plotting.neighbourhoods(data, clusters, hops=1, norm=True, save='scalar_fields_nhoods.svg')
    plotting.embedding(emb_MDS, beta, save='scalar_fields_MDS.svg') 

def f(x, alpha=0):
    return x[:,[0]]**2 - alpha*x[:,[1]]**2 


if __name__ == '__main__':
    sys.exit(main())