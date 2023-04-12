#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing
import matplotlib.pyplot as plt

# =============================================================================
# This example illustrates MARBLE for a scalar field on a flat surface
# =============================================================================

def main():
    
    #generate simple vector fields
    # f1: constant, f2: linear, f3: parabola, f4: saddle
    n = 512
    x = [dynamics.sample_2d(n, [[-1,-1],[1,1]], 'random') for i in range(4)]
    y = [f0(x[0]), f1(x[1]), f2(x[2]), f3(x[3])] #evaluated functions
        
    #construct PyG data object
    data = preprocessing.construct_dataset(x, y, graph_type='cknn', k=20)
    
    #train model
    params = {'epochs': 100, #optimisation epochs
              'order': 1, #order of derivatives
              'hidden_channels': 8, #number of internal dimensions in MLP
              'out_channels': 3,
              'include_self': False, #remove feature centers, for testing only, to get the figure in the SI
              'inner_product_features': False,
             }
    
    model = net(data, params=params)
    model.run_training(data)
    
    #evaluate model on data
    data = model.evaluate(data)
    n_clusters = 15 #use 15 clusters for simple visualisation
    data = postprocessing.distribution_distances(data, n_clusters=n_clusters)
    data = postprocessing.embed_in_2D(data)
    
    #plot
    titles=['Constant','Linear','Parabola','Saddle']
    plotting.fields(data, titles=titles, node_size=10, col=2)
    # plt.savefig('../results/fields.svg')
    plotting.embedding(data, data.y.numpy(), titles=titles, clusters_visible=True)
    # plt.savefig('../results/embedding.svg')
    plotting.histograms(data, titles=titles)
    # plt.savefig('../results/histogram.svg')
    plotting.neighbourhoods(data, hops=1, norm=True, figsize=(10, 20))
    # plt.savefig('../results/neighbourhoods.svg')
    
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