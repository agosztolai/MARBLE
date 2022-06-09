#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from GeoDySys.kernels import DD
from GeoDySys import utils
import matplotlib.pyplot as plt

n = 100
k=8
alpha = np.pi/4
 
np.random.seed(1)
x = np.random.uniform(low=(-1,-1),high=(1,1),size=(n,2))
xv, yv = np.meshgrid(np.linspace(-1,1,int(np.sqrt(n))), np.linspace(-1,1,int(np.sqrt(n))))
x = np.vstack([xv.flatten(),yv.flatten()]).T

#linear
def f1(x, alpha):
    return np.cos(alpha)*x[:,[0]] + np.sin(alpha)*x[:,[1]]
y = f1(x, alpha)

data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
K = DD(data, 'global')

der = np.hstack([np.matmul(K[0],y),np.matmul(K[1],y)])
derder = np.hstack([np.matmul(K[0],der[:,[0]]),np.matmul(K[1],der[:,[0]]),np.matmul(K[0],der[:,[1]]),np.matmul(K[1],der[:,[1]])])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 3),
                                subplot_kw={'aspect': 1})
ax1.scatter(x[:,0], x[:,1], c=y)
ax1.set_title(r'$(f_x,f_y)$')
ax1.axis('off')
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
ax2.scatter(x[:,0], x[:,1], c=y)
ax2.set_title(r'$f_{xx}$,$f_{yy}$')
ax2.axis('off')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax3.scatter(x[:,0], x[:,1], c=y)
ax3.set_title(r'$f_{xy}$,$f_{yx}$')
ax3.axis('off')
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
for ind in range(x.shape[0]):
    ax1.arrow(x[ind,0], x[ind,1], der[ind,0], der[ind,1], width=0.01)
    ax2.arrow(x[ind,0], x[ind,1], derder[ind,0], 0, width=0.01, color='r')
    ax2.arrow(x[ind,0], x[ind,1], 0, derder[ind,3], width=0.01, color='b')
    ax3.arrow(x[ind,0], x[ind,1], derder[ind,1], 0, width=0.01, color='r')
    ax3.arrow(x[ind,0], x[ind,1], 0, derder[ind,2], width=0.01, color='b')
    
PCM=ax1.get_children()[0] #get the mappable, the 1st and the 2nd are the x and y axes
plt.colorbar(PCM, ax=ax1)
plt.savefig('../results/kernel_linear.svg')

#parabola
def f2(x, alpha):
    return np.cos(alpha)*x[:,[0]]**2# - np.sin(alpha)*x[:,[1]]**2
y = f2(x, alpha)

data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
K = DD(data, 'global')

der = np.hstack([np.matmul(K[0],y),np.matmul(K[1],y)])
derder = np.hstack([np.matmul(K[0],der[:,[0]]),np.matmul(K[1],der[:,[0]]),np.matmul(K[0],der[:,[1]]),np.matmul(K[1],der[:,[1]])])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 3),
                                subplot_kw={'aspect': 1})
ax1.scatter(x[:,0], x[:,1], c=y)
ax1.set_title(r'$(f_x,f_y)$')
ax1.axis('off')
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
ax2.scatter(x[:,0], x[:,1], c=y)
ax2.set_title(r'$f_{xx}$,$f_{yy}$')
ax2.axis('off')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax3.scatter(x[:,0], x[:,1], c=y)
ax3.set_title(r'$f_{xy}$,$f_{yx}$')
ax3.axis('off')
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
for ind in range(x.shape[0]):
    ax1.arrow(x[ind,0], x[ind,1], der[ind,0], der[ind,1], width=0.01)
    ax2.arrow(x[ind,0], x[ind,1], derder[ind,0], 0, width=0.01, color='r')
    ax2.arrow(x[ind,0], x[ind,1], 0, derder[ind,3], width=0.01, color='b')
    ax3.arrow(x[ind,0], x[ind,1], derder[ind,1], 0, width=0.01, color='r')
    ax3.arrow(x[ind,0], x[ind,1], 0, derder[ind,2], width=0.01, color='b')
    
# PCM=ax1.get_children()[0] #get the mappable, the 1st and the 2nd are the x and y axes
# plt.colorbar(PCM, ax=ax1)
plt.savefig('../results/kernel_parabola.svg')
