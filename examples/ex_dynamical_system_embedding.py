#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from GeoDySys import plotting, solvers, time_series
from GeoDySys.geometry import furthest_point_sampling
# from torch_geometric.utils.convert import to_networkx
from scipy.spatial.transform import Rotation as R
import sys
          

def main():

    par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
    x0 = [-8.0, 7.0, 27.0]
    t = np.linspace(0, 20, 500)
    X = solvers.simulate_ODE('lorenz', t, x0, par)
    
    ax = None
    # ax = plotting.trajectories(X, style='->', lw=0.5, arrowhead=5, axis=False)
    
    N=50
    
    # ind, _ = furthest_point_sampling(X, N)
    
    # data = utils.construct_dataset(X, graph_type='cknn', k=20)
    # G = to_networkx(data, node_attrs=['pos'], edge_attrs=None, to_undirected=True,
    #         remove_self_loops=True)
    # ax = plotting.graph(G,node_values=None,show_colorbar=False,ax=ax,edge_alpha=0.3, edge_width=0.5)

    # ind = np.array(list(set(ind)))
    # _, nn = find_nn(ind, X, nn=2)
    
    # for i, nn_ in enumerate(nn):
    #     ax = circle(ax, 4, X[[ind[i]] + list(nn_)])
    
    # plt.savefig('../results/Lorenz_cover.svg')
    

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