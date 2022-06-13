#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from GeoDySys.solvers import generate_trajectories, sample_trajectories, simulate_ODE
from GeoDySys.time_series import delay_embed, find_nn
from GeoDySys import plotting, utils
from scipy.spatial.transform import Rotation as R
import sys
          

def main():

    par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
    x0 = [-8.0, 7.0, 27.0]
    t = np.linspace(0, 20, 1000)
    X = simulate_ODE('lorenz', t, x0, par)
    
    ax = plotting.trajectories(X, style='->', lw=0.5, arrowhead=5, axis=True)
    
    n=100
        
    x = np.random.uniform(np.min(X[:,0]),np.max(X[:,0]),n)
    y = np.random.uniform(np.min(X[:,1]),np.max(X[:,1]),n)
    z = np.random.uniform(np.min(X[:,2]),np.max(X[:,2]),n)
    
    ind = []
    for x_i in np.vstack([x,y,z]).T:
        ind.append(np.argmin(np.sum((X-x_i)**2,axis=1)))

    ind = np.array(list(set(ind)))
    _, nn = find_nn(ind, X, nn=2)
    
    for i, nn_ in enumerate(nn):
        ax = circle(ax, 4, X[[ind[i]] + list(nn_)])
        
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_zlim([0,50])
    
    plt.savefig('../results/Lorenz_cover.svg')
    
    # plt.figure()
    # plt.plot(t,X[:,0])
    
    # tau = -1
    # dim = 3
    
    # X = delay_embed(X[:,0],dim,tau)
    # plotting.trajectories(X, style='->', lw=0.5, arrowhead=5)
    # plt.savefig('../results/Lorenz_reconstructed.svg')
    

def circle(ax, r, X_p):
    
    theta = np.linspace(0, 2 * np.pi, 101)
    x = r*np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).T
    
    # ax.scatter(X_p[:,0],X_p[:,1],X_p[:,2],c='b')
    
    normal = np.cross((X_p[1]-X_p[0]), (X_p[2]-X_p[0]))
    # ax.scatter(normal[0],normal[1],normal[2],c='r')
    v_rot = np.cross(normal,[0,0,1])
    v_rot = np.divide(v_rot, np.sum(v_rot))
    # ax.scatter(v_rot[0],v_rot[1],v_rot[2],c='g')
    v_rot *= np.arccos(np.dot(v_rot, normal))
    M_R = R.from_rotvec(v_rot).as_matrix()
    x = np.matmul(M_R,x.T).T + X_p[0]
    
    ax.plot(x[:,0],x[:,1],x[:,2],'C1')
    
    return ax


if __name__ == '__main__':
    sys.exit(main())