import numpy as np
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
# import numpy.ma as ma
from .utils import parallel_proc

from itertools import product, combinations


# =============================================================================
# Embedding
# =============================================================================
def delay_embed(x, k, tau=-1, typ='asy'):
    """
    Delay-embedding for multi-dimensional time series x(t)

    Parameters
    ----------
    x : Txd array
        Multi-dimensional time series.
    k : int
        Embedding dimension. Needs to be at most 2*d+1, possibly smaller..
    tau : int, optional
        Delay parameter. The default is 1.
    typ : TYPE, optional
        Embedding based on points on one side (asymmetric) of both side of
        points (symmetric). The default is 'asy'.

    Returns
    -------
    Yd : (T-(k-1)*tau)*kd numpy array
        Time asymmetric or symmetric embedding. The default is 'asy'..

    """
    
    if len(x.shape)==1:
        x = x[:,None]
        
    dim = x.shape[1]
    
    #delay embed all dimensions
    Y = []
    for d in range(dim):
        Ytmp = delay_embed_scalar(x[:,d], k, tau, typ='asy')
        Y.append(Ytmp)
    
    #interleave dimensions
    shape = (Y[0].shape[0], Y[0].shape[1]*len(Y))
    # Y = np.vstack(Y).reshape(shape)
    Yd = np.empty(shape)
    for i in range(dim):
        Yd[:,i::dim]=Y[i]
    
    return Yd


def delay_embed_scalar(x, k, tau=-1, typ='asy'):
    """
    Delay-embedding for scalar time-series x(t).
    Builds prekictive coorkinates Y=[x(t), x(t-tau), ..., x(t-(k-1)*tau)]

    Parameters
    ----------
    x : Tx1 array
        Scalar time series.
    k : int
        Embedding dimension. Needs to be at most 2*d+1, possibly smaller.
    tau : int, optional
        Delay parameter. The default is 1.
    typ : 'asy', 'sym', optional
        Embedding based on points on one side (asymmetric) of both side of
        points (symmetric). The default is 'asy'.

    Returns
    -------
    Y : (T-(k-1)*tau)*k numpy array
        Delay embedded coorkinates.

    """
    
    #check if delay is in past (predictive embedding)
    if tau<0:
        tau = abs(tau)
        flip = True
    else:
        flip = False
        
    T = x.shape[0] #length of time series
    N = T - (k-1)*tau # length of embedded signal
    Y = np.zeros([N,k])
        
    if typ == 'asy':  
        for ki in range(k):
            ind = np.arange(ki*tau, N+ki*tau)
            Y[:,ki] = x[ind]
            
        if flip:
            Y = np.flip(Y, axis=1)
            
    elif typ == 'sym' and k % 2 == 0:
        for ki in range(-k//2,k//2):
            ind = np.arange((k//2+ki)*tau, N+(k//2+ki)*tau)
            Y[:,ki] = x[ind]
            
    elif typ == 'sym' and k % 2 == 1:    
        for ki in range(-(k-1)//2,(k-1)//2+1):
            ind = np.arange(((k-1)//2+ki)*tau, N+((k-1)//2+ki)*tau)
            Y[:,ki] = x[ind]
            
    return Y

# =============================================================================
# Neighbour search
# =============================================================================
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
    res = parallel_proc(nb_query, 
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


# =============================================================================
# Projection
# =============================================================================
def random_projection(X, dim_out=1, seed=1):
    """
    Randomly project dynamical system to a low dimensional plane

    Parameters
    ----------
    X : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space.
    dim_out : int, optional
        Dimension of the projection plane. The default is 1 (scalar time 
                                                             series).

    Returns
    -------
    x_proj : nxdim_out array
        Projected dynamics.

    """
    
    from scipy.stats import special_ortho_group
    from numpy.random import RandomState
    
    dim = X.shape[1]
    rs = RandomState(seed)
    
    R = special_ortho_group.rvs(dim, random_state=rs)
    
    x_proj = np.matmul(R, X.T).T
    x_proj = x_proj[:,:dim_out]
    
    return x_proj


# =============================================================================
# Prediction
# =============================================================================
def predict(X, t, nn=2, ind=None, n_jobs=2):
    """
    Compute the zeroth-order prediction by the simplex projection method.
    
    Parameters
    ----------
    X : nxdim matrix of attractors points.
    t : int or array of prediction horizons
    nn : number of nearest neighbours to use.
    ind : indices of initial points. If None, then all points are used up to
        time n-t.
    n_jobs : Number of CPUs used for the nearest neighbour search.
    
    Returns
    -------
    X_pred : len(ind)xdim matrix of predicted points
    error : len(x) array of prediction errors

    """
    
    assert nn>1, 'At least two neighbours are needed!'
    
    if isinstance(t, int):
        t = [t]
    
    if ind is None:
        ind = np.arange(X.shape[0]-np.max(t)-1)
    
    #find neighbours
    dist, n_inds = find_nn(ind, X[ind], nn=nn, n_jobs=n_jobs)
    
    X_pred, error = [], []
    for t_ in t:
        X_pred_, error_ = simplex_projection(X, t_, n_inds, dist)
        X_pred.append(X_pred_)
        error.append(error_)
        
    X_pred = np.stack(X_pred, axis=1)
    error = np.stack(error, axis=1)
        
    return X_pred, error


def simplex_projection(X, t, n_inds, dist=None):
    """
    Helper function of predictions(). Implements Lorenz’s “method of analogs.
    Takes points X[n_ids], pushes them forward by time t and averages them.

    """
    
    n = len(n_inds)
    
    #average neighbours weighted by the distances
    error, X_pred = np.zeros(n), np.zeros([n, X.shape[1]])
    for i, n_ind in enumerate(n_inds):
        X_pred[i] = np.mean(X[n_ind+t]*dist[i][:, None], axis=0)
        error[i] = np.linalg.norm(X[i] - X_pred)
        
    return X_pred, error


# =============================================================================
# Multi-view embedding
# =============================================================================
def multi_view_embedding(x, dim, tau=-1):
    
    assert isinstance(x, list), 'Input must be a list'
    
    Y_mv = valid_embeddings(x, dim, tau)    
    # rank = rank_embeddings(Y_mv)
            
    return Y_mv


def valid_embeddings(x, dim, tau):
    
    #embed all time series individually
    Y_nodelay, Y_delay = [], []
    for i in range(len(x)):
        Y_tmp = delay_embed(x[i], dim, tau)
        Y_tmp = standardize(Y_tmp)
        
        #separate coordinate without delay and with delay
        Y_nodelay += list(Y_tmp[:,0].T)
        Y_delay += list(Y_tmp[:,1:].T)
        
    Y_nodelay = np.vstack(Y_nodelay).T
    Y_delay = np.vstack(Y_delay).T
    
    #all combinations on valid embeddings
    n1 = Y_nodelay.shape[1]
    n2 = Y_delay.shape[1]
    s = []
    for i in range(dim):
        s += list(product(comb_(n1,dim-i), comb_(n2,i)))
    
    #stack valid embeddings
    Y_mv = []
    for a, b in s:
        if len(b)==0:
            Y_mv.append(Y_nodelay.T[list(a)].T)
        else:
            print(list(a))
            print(list(b))
            Y_mv.append(np.hstack([Y_nodelay[:,list(a)],
                                   Y_delay[:,list(b)]]))
        
    return Y_mv


def rank_embeddings(Y):

    rank = 0
    return rank


def multi_view_prediction(Y, n, t):
    
    if not isinstance(Y, list):
        Y = [Y]
        
    count = 0
    ypred = 0
    for Y_ in Y:
        dist = np.norm(Y_ - Y_[n], axis=1)
        ind = np.argmin(dist) + t
        
        if len(Y_) < ind-1:
            ypred += Y_[ind]
            count += 1
        
    return ypred/count


def comb_(n,r):
    return combinations(range(n), r=r)


def standardize(X):
    
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True)
    
    return X


# def valid_flows(t_ind, ts, T):
#     """
#     Mask out invalid trajectories.

#     Parameters
#     ----------
#     t_ind : list[int]
#         Time indices corresponding to the sampled short trajectories.
#     ts : list[int]
#         Start of trajectory.
#     T : int or list[int]
#         End of trajectory or time horizon.

#     Returns
#     -------
#     tt : list[int]
#         Start of trajectory.
#     ts : list[int]
#         End of trajectory.

#     """
    
#     if isinstance(T, int):
#         tt = [ts[i]+T for i in range(len(ts))]
#     else:
#         tt = T
#     assert len(tt)==len(ts), 'Number of source points must equal to the \
#             number of target points.'
            
#     t_breaks = np.zeros_like(t_ind)
#     t_breaks[np.array(t_ind)==0] = 1
#     t_breaks[0] = 0
    
#     invalid = np.zeros_like(tt)
#     for i,(s,t) in enumerate(zip(ts,tt)):
#         if t>len(t_ind)-2 or s<0 or t<=s or np.sum(t_breaks[s:t+1])>0:
#             invalid[i] = 1
        
#     ts = ma.array(ts, mask=invalid)
#     tt = ma.array(tt, mask=invalid)
        
#     return ts, tt


# def generate_flow(X, ts, T):
#     """
#     Obtain trajectories of between timepoints.

#     Parameters
#     ----------
#     X : np array
#         Trajectories.
#     ts : int or np array or list[int]
#         Source timepoint.
#     T : int or list[int]
#         End of trajectory or time horizon.

#     Returns
#     -------
#     X_sample : list[np array].
#         Set of flows of length T.

#     """
    
#     ts = ma.array(ts, dtype=int)
    
#     if isinstance(T, int):
#         tt = ma.array([ts[i]+T for i in range(len(ts))])
#         tt = ma.array(tt, mask=ts.mask, dtype=int)
#     else:
#         tt = ma.array(T)
#         assert len(tt)==len(ts), 'Number of source points must equal to the \
#             number of target points.'
    
#     X_sample = []
#     for s,t in zip(ts,tt):
#         if not ma.is_masked(s) and not ma.is_masked(t):
#             X_sample.append(X[s:t+1])

#     return X_sample, ts[~ts.mask], tt[~tt.mask]