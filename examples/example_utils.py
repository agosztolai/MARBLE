import numpy as np
import torch
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from MARBLE import utils, geometry, plotting
from RNN_scripts import dms

"""Some functions that are used for the exampels"""

def circle(ax, r, X_p, col='C1'):
    
    theta = np.linspace(0, 2 * np.pi, 101)
    x = r*np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).T
    
    # ax.scatter(X_p[:,0],X_p[:,1],X_p[:,2],c='b')
    
    if X_p.shape[1]==3:
        normal = np.cross((X_p[1]-X_p[0]), (X_p[2]-X_p[0]))
        # ax.scatter(normal[0],normal[1],normal[2],c='r')
        v_rot = np.cross(normal,[0,0,1])
        v_rot = np.divide(v_rot, np.sum(v_rot))
        # ax.scatter(v_rot[0],v_rot[1],v_rot[2],c='g')
        v_rot *= np.arccos(np.dot(v_rot, normal))
        M_R = R.from_rotvec(v_rot).as_matrix()
        x = np.matmul(M_R,x.T).T + X_p[0]
    
        ax.plot(x[:,0],x[:,1],x[:,2],col)
    elif X_p.shape[1]==2:
        x = x[:,:2] + X_p[0]
        ax.plot(x[:,0], x[:,1],col)
    else:
        NotImplementedError
    
    return ax


def find_nn(ind_query, X, nn=1, r=None, theiler=10, n_jobs=2):
    """
    Find nearest neighbors of a point on the manifold

    Parameters
    ----------
    ind_query : 2d np array, list[2d np array]
        Index of points whose neighbors are needed.
    x : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space.
    nn : int, optional
        Number of nearest neighbors. The default is 1.
    theiler : int, optional
        Theiler exclusion. Do not include the points immediately before or 
        after in time the query point as neighbours.
    n_jobs : int, optional
        Number of processors to use. The default is 2.
        
    Returns
    -------
    dist : list[list]
        Distance of nearest neighbors.
    ind : list[list]
        Index of nearest neighbors.

    """
        
    if isinstance(ind_query, list):
        ind_query = np.vstack(ind_query)
    
    #Fit neighbor estimator object
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    
    inputs = [kdt, X, ind_query, r, nn, theiler]
    res = utils.parallel_proc(nb_query, 
                        range(len(ind_query)), 
                        inputs, 
                        desc="Computing neighbours...")
    
    dist, ind = zip(*res)
    
    return dist, ind


def nb_query(inputs, i):
    
    kdt, X, ind_query, r, nn, theiler  = inputs
    
    x_query = X[ind_query][[i]]
    ind_query = ind_query[i]
    if r is not None:
        ind, dist = kdt.query_radius(x_query, r=r, return_distance=True, sort_results=True)
        ind = ind[0]
        dist = dist[0]
    else:
        # apparently, the outputs are reversed here compared to query_radius()
        dist, ind = kdt.query(x_query, k=nn+2*theiler+1)
        
    #Theiler exclusion (points immediately before or after are not useful neighbours)
    dist = dist[np.abs(ind-ind_query)>theiler][:nn]
    ind =   ind[np.abs(ind-ind_query)>theiler][:nn]
            
    return dist, ind


def reject_outliers(*args, min_v=-3, max_v=3):
    inds = []
    for arg in args:
        inds.append(np.where((arg>min_v).all(1)*(arg<max_v).all(1))[0])
        
    return list(set.intersection(*map(set,inds)))


def initial_conditions(n, reps, area = [[-3,-3],[3,3]], seed=0):
    X0_range = [geometry.sample_2d(n, area, 'random', seed=i+seed) for i in range(reps)]
        
    return X0_range


def plot_phase_portrait(pos, vel, ax=None, node_feature=None, style='>', lw=2, scale=1., spacing=1):
    if not isinstance(pos, list):
        pos = [pos]
    if not isinstance(vel, list):
        vel = [vel]

    if node_feature is None:
        node_feature = len(pos) * [None]
        
    for p, v, nf in zip(pos, vel, node_feature):
        ax = plotting.trajectories(p, v, ax=ax, style=style, node_feature=nf, lw=lw, scale=scale, axis=False, alpha=1., arrow_spacing=spacing)
        
    return ax


def spiking_data(file = '../data/conditions_spiking_data.mat'):
    
    #data is a matrix with shape (trials, conditions)
    data = loadmat(file)['result']
    
    conditions=['DownLeft','DownRight','Left','Right','UpLeft','UpRight','Up']
    
    rates = {}
    for i, cond in enumerate(conditions):
                
        rates_trial = []
        for t, trial in enumerate(data[:,i]):
            if trial.shape[1] != 0:                
                try:
                    rates_channel = []
                    #trial[0][0] is a matrix with shape (channel, timesteps)
                    for c, channel in enumerate(trial[0][0]):
                        spikes = np.where(channel)[0]
                        
                        st = neo.SpikeTrain(spikes, units='ms', t_stop=1200)
                        r = instantaneous_rate(st, sampling_period=50*ms).magnitude

                        rates_channel.append(r.flatten())
                                                                    
                    rates_trial.append(np.vstack(rates_channel))
                except:
                    continue
            
        #rates[cond] is a matrix with shape (trials, channels, timesteps)
        rates[cond] = np.stack(rates_trial, axis=0)
        
    pickle.dump(rates, open('../data/rate_data.pkl','wb'))
    
    
def generate_trajectories(net, input, epochs, n_traj):
    
    traj = []
    for i in range(len(input)):
        conds = []
        for k in range(n_traj):
            #net.h0.data = torch.rand(size=net.h0.data.shape) #random ic
            _, traj_ = net(input[i].unsqueeze(0))
            traj_ = traj_.squeeze().detach().numpy()
            traj_epoch = [traj_[e:epochs[j+1]] for j, e in enumerate(epochs[:-1])]
            conds.append(traj_epoch)
            
        traj.append(conds)
        
    return traj, input


def plot_experiment(net, input, traj, epochs, traj_to_show=1):
    fig, ax = plt.subplots(4, 5, figsize=(25, 20))
    idx = np.floor(np.linspace(0, len(input)-1, 4))
    for i in range(4):
        for j, e in enumerate(epochs[:-1]):
            dms.plot_field(net, input[int(idx[i]), e], ax[i][j], sizes=1.3)
            epoch = [c[j] for c in traj[i]]
            dms.plot_trajectories(net, epoch, ax[i][j], c='#C30021', n_traj=traj_to_show)
            if j>0:
                epoch = [c[j-1] for c in traj[i]]
                dms.plot_trajectories(net, epoch, ax[i][j], c='#C30021', style='--', n_traj=traj_to_show)
    
    ax[0][0].set_title('Fix')
    ax[0][1].set_title('Stim 1')
    ax[0][2].set_title('Delay')
    ax[0][3].set_title('Stim 2')
    ax[0][4].set_title('Decision')
    
    for i in range(4):
        ax[i][0].set_ylabel('$\kappa_2$')
    for i in range(5):
        ax[3][i].set_xlabel('$\kappa_1$')
        
    fig.subplots_adjust(hspace=.1, wspace=.1)