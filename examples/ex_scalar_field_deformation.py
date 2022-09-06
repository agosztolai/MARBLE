#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from GeoDySys import plotting, utils, geometry, net

def main():
    
    #parameters
    k = 15
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
    n_steps = 3
    alpha = np.linspace(1, 0.75, n_steps)
    nr = 10
    ntheta = 40
    x_ = np.linspace(-np.pi, np.pi, nr)
    y_ = np.linspace(-np.pi, np.pi, ntheta)
    x_, y_ = np.meshgrid(x_, y_)
    X = np.column_stack([x_.flatten(), y_.flatten()])
    
    y = [f1(X) for i in range(n_steps)] + [f2(X) for i in range(n_steps)]
    x = 2 * [sample_cone(a, nr, ntheta) for a in alpha]
    
    # ind = geometry.furthest_point_sampling(x[0], N=200)[0]
    # x = [x_[ind] for x_ in x]
    # y = [y_[ind] for y_ in y]
    
    #construct PyG data object
    data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
    
    #train model
    model = net(data, gauge='global', **par)
    model.run_training(data)
    model.evaluate(data)
    emb, clusters, dist = model.cluster_and_embed(n_clusters=n_clusters)
    emb_MDS = geometry.embed(dist, embed_typ='MDS')
    
    #plot
    titles = [r'$f_1,\alpha$ = ' + str(a) for a in alpha] + \
        [r'$f_2,\alpha$ = ' + str(a) for a in alpha]
    plotting.fields(data, titles=titles, col=3, figsize=(8,5), save='scalar_fields.svg')
    plotting.embedding(emb, data.y.numpy(), clusters, titles=titles, save='scalar_fields_embedding.svg') 
    plotting.histograms(clusters, col=3, figsize=(13,3), titles=titles, save='scalar_fields_histogram.svg')
    plotting.neighbourhoods(data, clusters, hops=1, norm=True, save='scalar_fields_nhoods.svg')
    labels = np.array([0]*n_steps + [1]*n_steps)
    plotting.embedding(emb_MDS, labels, save='scalar_fields_MDS.svg') 

def f1(x):
    return np.cos(x[:,[0]]) + np.sin(x[:,[1]])

def f2(x):
    return np.cos(x[:,[0]]) + np.sin(2*x[:,[1]])

def sample_cone(alpha, nr, ntheta):
    r = np.sqrt(np.linspace(0.5, 5, nr))
    theta = np.linspace(0, 2*np.pi, ntheta)
    r, theta = np.meshgrid(r, theta)
    X = r*np.cos(theta)
    Y = r*np.sin(theta)
    Z = -(alpha*r)**2
    
    return np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])


if __name__ == '__main__':
    sys.exit(main())