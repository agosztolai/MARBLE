#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import sys
from GeoDySys import utils, geometry, plotting
import seaborn as sns
import matplotlib.pyplot as plt

def main():

    n = 100
    k=8
    alpha = np.pi/4
     
    # np.random.seed(1)
    xv, yv = np.meshgrid(np.linspace(-1,1,int(np.sqrt(n))), np.linspace(-1,1,int(np.sqrt(n))))
    x = np.vstack([xv.flatten(),yv.flatten()]).T
    
    #evaluate functions
    n_steps = 5
    alpha = np.around(np.linspace(-.5, .5, n_steps), decimals=2)
    nr = 200
    ntheta = 200
    x_ = np.linspace(-np.pi, np.pi, nr)
    y_ = np.linspace(-np.pi, np.pi, ntheta)
    x_, y_ = np.meshgrid(x_, y_)
    X = np.column_stack([x_.flatten(), y_.flatten()])
    
    #transfer function to cone
    y = [f1(X) for i in range(n_steps)]
    x = [sample_cone(a, nr, ntheta) for a in alpha]
    
    ind, _ = geometry.furthest_point_sampling(x[0], stop_crit=0.025)
    x = [x_[ind] for x_ in x]
    y = [y_[ind] for y_ in y]
    
    #construct PyG data object
    data = [utils.construct_dataset(x[i], y[i], graph_type='cknn', k=k) for i in range(n_steps)]
    
    #compute gradient operator
    gauges, _ = geometry.compute_gauges(data[0], local=False)
    K = [geometry.DD(data[i].pos, data[i].edge_index, gauges) for i in range(n_steps)]
    
    data = [data[i].to_data_list()[0] for i in range(n_steps)]
    
    der = [np.hstack([np.matmul(K[i][0],y[i]),np.matmul(K[i][1],y[i]),np.matmul(K[i][2],y[i])]) for i in range(n_steps)]
    der = [np.linalg.norm(der[i], axis=1, keepdims=True) for i in range(n_steps)]
        
    for i in range(n_steps):
        data[i].x = torch.tensor(der[i]) 
        
    plotting.fields(data, col=n_steps, figsize=(8,5))
    
    #plot differences in deformations
    e = [abs(der[i] - der[len(der)//2])/abs(der[len(der)//2]) for i in range(n_steps)]
    
    plt.figure()
    sns.boxplot(data=e, color='C0', showfliers = False)
    plt.xticks(ticks=np.arange(n_steps), labels=alpha)
    plt.ylabel('Fractional error in gradient')
    plt.xlabel(r'$\alpha$')


def f1(x):
    return np.cos(x[:,[0]]) + np.sin(x[:,[1]])

def sample_cone(alpha, nr, ntheta):
    r = np.sqrt(np.linspace(0.5, 5, nr))
    theta = np.linspace(0, 2*np.pi, ntheta)
    r, theta = np.meshgrid(r, theta)
    X = r*np.cos(theta)
    Y = r*np.sin(theta)
    Z = -alpha*r**2
    
    return np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

if __name__ == '__main__':
    sys.exit(main())