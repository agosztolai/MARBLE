#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
# from math import comb
import random
from GeoDySys import time_series, curvature, plotting, solvers, discretisation, op_calculations
random.seed(a=0)
            
"""1. Simulate system"""
# x0 = [0.1,0.1] 
# par = {'mu': 0.1}
# fun = 'vanderpol'

par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
fun = 'lorenz'

# par['sigma']*((par['sigma']+par['beta']+3)/(par['sigma']-par['beta']-1))

n=2000
x0 = [-8.0, 7.0, 27.0]
t = np.linspace(0, 500, n)
t_ind = np.arange(n)
mu, sigma = 0, 1 # mean and standard deviation
X = solvers.simulate_ODE(fun, t, x0, par, noise=False, mu=mu, sigma=sigma)
t_sample = t_ind


"""2. Random project and then delay embed"""
# x = time_series.random_projection(X, seed=0)
# dim = 3
# X = time_series.delay_embed(X[:,0],dim,tau=-1)
# t_sample = t_sample[:-dim]


"""3. Discretisation"""
centers, sizes, labels = discretisation.kmeans_part(X, 100)
# centers, sizes, labels = discretisation.maxent_part(X, dim, .1)

"""4. Markov transition operator"""
P = op_calculations.get_transition_matrix(t_ind,labels,T=5)
 
"""5. Curvature matrix"""
K = curvature.get_curvature_matrix(X,t_ind,labels,T=5,Tmax=20)

# K = np.clip(K, -0.1, 0.1)

"""Plotting"""
# ax = plotting.trajectories(X, color=None, style='o', lw=1, ms=1)

#discretisation
ax = plotting.discretisation(centers, sizes, ax=None, alpha=0.2)
ax = plotting.transition_diagram(centers, P.todense(), ax=ax, radius=10, dim=3, lw=2, ms=1, alpha=1)

#transition matrix
# plt.figure()
# plt.imshow(P.todense())

# plt.figure()
# plt.imshow(K)
#
#curvature across time horizons
# plot.plot_curvatures(times,kappas,ylog=True)
