import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R

from MARBLE import utils


"""Some functions that are used for the exampels"""

def circle(ax, r, X_p):
    
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
    
        ax.plot(x[:,0],x[:,1],x[:,2],'C1')
    elif X_p.shape[1]==2:
        x = x[:,:2] + X_p[0]
        ax.plot(x[:,0], x[:,1],'C1')
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