#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing
import matplotlib.pyplot as plt

# =============================================================================
# This example illustrates MARBLE for a vector field on a flat surface
# =============================================================================

def main():
    
    #generate simple vector fields
    # f0: linear, f1: point source, f2: point vortex, f3: saddle
    n = 512
    x = [dynamics.sample_2d(n, [[-1,-1],[1,1]], 'random') for i in range(4)]
    y = [f0(x[0]), f1(x[1]), f2(x[2]), f3(x[3])] #evaluated functions
        
    #construct PyG data object
    data = preprocessing.construct_dataset(x, 
                                           y, 
                                           graph_type='cknn', 
                                           k=20)
    
    #train model
    params = {'epochs': 50, #optimisation epochs
              'order': 1, #first-order derivatives are enough because the vector field have at most first-order features
              'hidden_channels': 16, #16 is enough in this simple example
              'out_channels': 3, #3 is enough in this simple example
              'inner_product_features': True, #try changing this to False and see how the embeddings change
              }
    model = net(data, params=params)
    model.run_training(data)
    
    #evaluate model on data
    data = model.evaluate(data)
    n_clusters = 15 #use 15 clusters for simple visualisation
    data = postprocessing.distribution_distances(data, n_clusters=n_clusters, cluster_typ='kmeans')
    data = postprocessing.embed_in_2D(data)
    
    #plot results
    titles=['Linear left','Linear right','Vortex right','Vortex left']
    plotting.fields(data, titles=titles, col=2)
    # plt.savefig('../results/fields.svg')
    plotting.embedding(data, data.y.numpy(),titles=titles, clusters_visible=True)
    # plt.savefig('../results/embedding.svg')
    plotting.histograms(data, titles=titles)
    # plt.savefig('../results/histogram.svg')
    plotting.neighbourhoods(data)
    # plt.savefig('../results/neighbourhoods.svg')
    
def f0(x):
    return x*0 + np.array([-1,-1])

def f1(x):
    return x*0 + np.array([1,1])

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


if __name__ == '__main__':
    sys.exit(main())