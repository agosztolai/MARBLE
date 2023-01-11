#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from MARBLE import geometry
from MARBLE import utils
import matplotlib.pyplot as plt
from MARBLE.layers import AnisoConv

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
y = torch.tensor(y)

data = utils.construct_dataset(x, y, graph_type='cknn', k=k)
gauges, _ = geometry.compute_gauges(data[0], local=False)

#generate random matrices and rotate gauges arbitrarily
import numpy as np
import torch
np.random.seed(0)
Rot = []
for i, ga in enumerate(gauges):
    t = np.random.uniform(low=0,high=2*np.pi)
    R = np.array([[np.cos(t), -np.sin(t)], 
                        [np.sin(t),  np.cos(t)]])
    R = torch.tensor(R, dtype=torch.float32)
    Rot.append(R)
    gauges[i] = R@ga
    
y = geometry.coordinate_transform(y,gauges)

K = geometry.gradient_op(data.pos, data.edge_index, gauges)
grad = AnisoConv()
der = grad(y, data.edge_index, (len(y),len(y)), K)
# rotate derivatives to match the ambient coordinate system
for i, R in enumerate(Rot):
    R = torch.tensor(R, dtype=torch.float64)
    der[i] = R@der[i]
derder = grad(der, data.edge_index, (len(y),len(y)), K)
for i, R in enumerate(Rot):
    R = torch.tensor(R, dtype=torch.float64)
    derder[i] = torch.einsum('ij,kj->ik', R, derder[i].view(2,2)).view(1,4)

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
# plt.savefig('../results/kernel_linear.svg')

#parabola
def f2(x, alpha):
    return np.cos(alpha)*x[:,[0]]**2# - np.sin(alpha)*x[:,[1]]**2
y = f2(x, alpha)
y = torch.tensor(y)

der = grad(y, data.edge_index, (len(y),len(y)), K)
# for i, R in enumerate(Rot):
#     R = torch.tensor(R, dtype=torch.float64)
#     der[i] = R@der[i]
derder = grad(der, data.edge_index, (len(y),len(y)), K)
# for i, R in enumerate(Rot):
#     R = torch.tensor(R, dtype=torch.float64)
#     derder[i] = torch.einsum('ij,kj->ik', R, derder[i].view(2,2)).view(1,4)

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
# plt.savefig('../results/kernel_parabola.svg')

#linear directional average
def f1(x, alpha):
    return np.cos(alpha)*x[:,[0]] + np.sin(alpha)*x[:,[1]]
y = f1(x, alpha)
