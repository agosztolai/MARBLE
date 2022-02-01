#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
# from math import comb
import random
from GeoDySys import time_series, curvature, plotting, solvers, discretisation, op_calculations
random.seed(a=0)
import sys

def main():
    """1. Simulate system"""
    # x0 = [0.1,0.1] 
    # par = {'mu': 0.1}
    # fun = 'vanderpol'
    
    par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
    fun = 'lorenz'
    
    # par = {'a': 0.15, 'b': 0.2, 'c': 10.0}
    # fun = 'rossler'
    
    # par['sigma']*((par['sigma']+par['beta']+3)/(par['sigma']-par['beta']-1))
    
    tn=1000
    x0 = [-8.0, 7.0, 27.0]
    t = np.linspace(0, 20, tn)
    t_ind = np.arange(tn)
    mu, sigma = 0, 0. # mean and standard deviation of additive noise
    X = solvers.simulate_ODE(fun, t, x0, par, mu=mu, sigma=sigma)
    t_sample = t_ind
    
    
    """2. Random project and then delay embed"""
    # x = time_series.random_projection(X, seed=0)
    # dim = 3
    # X = time_series.delay_embed(X[:,0],dim,tau=-1)
    # t_sample = t_sample[:-dim]
    
    
    """3. Compute curvature of trajectories starting at t_sample"""
    # n_sample=200
    # t_sample = random.sample(list(np.arange(tn)), n_sample)
    
    T=5
    kappas = curvature.curvature_trajectory(X,t_ind,t_sample,T)
    kappas = np.clip(kappas, -0.1, 0.1)
    
    """4. Plotting"""
    #curvature across time horizons
    # plot.plot_curvatures(times,kappas,ylog=True)
    
    #plotting some trajectories and their neighbours
    #find nearest neighbours
    _, nn = time_series.find_nn(t_sample, X, nn=5)
    t_nn = np.hstack([np.array(t_sample)[:,None],np.array(nn)])
    ts, tt = time_series.valid_flows(t_ind, t_nn.flatten(), T)
    ts = ts.reshape(t_nn.shape)
    tt = tt.reshape(t_nn.shape)
    ax = plotting.trajectories(X, node_feature=None, style='o', lw=1, ms=1)
    flows_n, _, _ = time_series.generate_flow(X, ts[55,1:], T)
    plotting.trajectories(np.vstack(flows_n), ax=ax, node_feature='C1', style='-', lw=1, ms=4)
    flow, _, _ = time_series.generate_flow(X, ts[55,[0]], T)
    plotting.trajectories(np.vstack(flow), ax=ax, node_feature='C3', style='-', lw=1, ms=4)
    
    #plotting all curvatures
    ax = plotting.trajectories(X, node_feature=kappas, style='o', lw=0.5, ms=6)
    # plt.savefig('../results/manifold.svg')
    
    #plot trajectory with curvature values
    #use coordinate 0 for lorenz 2 for rossler
    ax = plotting.time_series(t,X[:,0], node_feature=kappas, style='-', lw=2)
    # ax.set_xlim([0,20])

if __name__ == '__main__':
    sys.exit(main())