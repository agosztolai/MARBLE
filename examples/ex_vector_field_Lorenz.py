#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from MARBLE import plotting, utils, net, postprocessing
from DE_library import simulate_trajectories
import matplotlib.pyplot as plt


def main():
    
    #parameters
    n_clusters = 15
    
    par = {'batch_size': 128,
           'epochs': 100, #optimisation epochs
           'order': 1, #order of derivatives
           'n_lin_layers': 2,
           'hidden_channels': 16, #number of internal dimensions in MLP
           'out_channels': 4,
           'inner_product_features': False,
           'autoencoder': False
           }
    
    #simulate system
    X0_set = [[[-8., 7., 27.]], [[6, -6., 28.]]]
    t = np.linspace(0, 10, 250)
    
    s, b = 10, 8/3
    rh = s*(s+b+3)/(s-b-1)
    rho = np.linspace(rh-2, rh+2, 2)

    X, V = [], []
    for r in rho:
        X_tmp, V_tmp = [], []
        for i, X0 in enumerate(X0_set):
            p, v = simulate_system(t, X0, rho=r)
                            
            X_tmp.append(np.vstack(p)[10:]) #eliminate transient
            V_tmp.append(np.vstack(v)[10:])
            
        X.append(np.vstack(X_tmp))
        V.append(np.vstack(V_tmp))
        
    #construct PyG data object
    data = utils.construct_dataset(X, features=V, graph_type='cknn', k=10, stop_crit=0.02)
    
    #train model
    model = net(data, **par)
    model.run_training(data)
    
    #evaluate model on data
    data = model.evaluate(data)
    data = postprocessing(data, n_clusters=n_clusters, cluster_typ='kmeans')
    
    #plot
    titles=[r'$\rho={}$'.format(r) for r in rho]
    plotting.fields(data, axshow=True, plot_gauges=False, titles=titles, figsize=(8,8), col=2, scale=3, width=10)
    # plt.savefig('../results/fields.svg')
    plotting.embedding(data, data.y.numpy(),titles=titles)
    # plt.savefig('../results/embedding.svg')
    plotting.histograms(data, titles=titles)
    # plt.savefig('../results/histogram.svg')
    plotting.neighbourhoods(data)
    # plt.savefig('../results/neighbourhoods.svg')
    

def simulate_system(t, X0, rho=28.0):
    p, v = simulate_trajectories('lorenz', X0, t, par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': rho, 'tau': 1.0})
    pos, vel = [], []
    for p_, v_ in zip(p,v):
        pos.append(p_)
        vel.append(v_)
        
    return pos, vel
    

if __name__ == '__main__':
    sys.exit(main())