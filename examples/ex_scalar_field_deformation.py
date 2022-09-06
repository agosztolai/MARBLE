#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from GeoDySys import plotting, utils, geometry, net

def main():
    
    #parameters
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
    n_steps = 5
    alpha = np.linspace(1, 0, n_steps)
    nr = 10
    ntheta = 20
    x_ = np.linspace(-np.pi, np.pi, nr)
    y_ = np.linspace(-np.pi, np.pi, ntheta)
    x_, y_ = np.meshgrid(x_, y_)
    X = np.column_stack([x_.flatten(), y_.flatten()])
    
    y = [f(X) for i in range(n_steps)]
    x = [sample_cone(a) for a in alpha]
    
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
    
    #train model
    model = net(data, gauge='global', **par)
    model.run_training(data)
    model.evaluate(data)
    emb, clusters, dist = model.cluster_and_embed(n_clusters=n_clusters)
    emb_MDS = geometry.embed(dist, embed_typ='MDS')
    
    #plot
    plotting.fields(data, col=3, figsize=(10,5), save='scalar_fields.svg')
    plotting.embedding(emb, data.y.numpy(), clusters, save='scalar_fields_embedding.svg') 
    plotting.histograms(clusters, col=5, figsize=(13,3), save='scalar_fields_histogram.svg')
    plotting.neighbourhoods(data, clusters, hops=1, norm=True, save='scalar_fields_nhoods.svg')
    plotting.embedding(emb_MDS, alpha, save='scalar_fields_MDS.svg') 


def f(x):
    return np.cos(x[:,[0]]) + np.sin(x[:,[1]])

def sample_cone(alpha, nr=10, ntheta=20):
    r = np.sqrt(np.linspace(0.5, 5, nr))
    theta = np.linspace(0, 2*np.pi, ntheta)
    r, theta = np.meshgrid(r, theta)
    X = r*np.cos(theta)
    Y = r*np.sin(theta)
    Z = -(alpha*r)**2
    
    return np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])


if __name__ == '__main__':
    sys.exit(main())