#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:25:47 2022

@author: gosztola
"""

import numpy as np
from DE_library import simulate_ODE, simulate_phase_portrait, plotting
import matplotlib.pyplot as plt

import sys
from MARBLE import utils, geometry, net

t0, t1, dt = 0, 5, 0.05
t = np.arange(t0, t1, dt)
n = 100

X0_range = [geometry.sample_2d(n, [[-1,-1],[1,1]], 'random') for i in range(4)]

fig, ax = plt.subplots(1,4, figsize=(20,5))

def simulate_system(beta1, X0_range):
    pos, vel = simulate_phase_portrait('bogdanov_takens', t, X0_range, par = {'beta1': -0.15, 'beta2': -0.6})
    return pos, vel

def plot_phase_portrait(pos, vel, ax):

    for p, v in zip(pos, vel):
        ax = plotting.trajectories(p, v, ax=ax, style='->', lw=1, arrowhead=.03, arrow_spacing=7, axis=False, alpha=None)
    ax.axis('square')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    
pos, vel = simulate_system(-0.15, X0_range[0])
plot_phase_portrait(pos, vel, ax[0])