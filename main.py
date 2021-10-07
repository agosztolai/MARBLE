import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull


"""
We have the following functions:
    1) delay_embed, delay_embed_scalar: delay embedding (scalar and multi-d time series)
    2) standardize_data: standardize data
    3) find_nn: find nearest neighbors to a point
    4) geodesic_dist: geodesic distance between two points (with and without smoothing spline)
    5) random_projection: obtain scalar time series by random projections
"""

def delay_embed(x, k, tau=1, typ='asy'):
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


def standardize_data(X, axis=0):
    """
    Normalize data

    Parameters
    ----------
    X : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space..
    axis : 0,1, optional
        Dimension to normalize. The default is 0 (along dimensions).

    Returns
    -------
    X : nxd array (dimensions are columns!)
        Normalized data.

    """
    
    X -= np.mean(X, axis=axis, keepdims=True)
    X /= np.std(X, axis=axis, keepdims=True)
        
    return X


def find_nn(x_query, X, nn=1, nmax=10, n_jobs=-1):
    """
    Find nearest neighbors of a point on the manifold

    Parameters
    ----------
    x_query : 2d np array, list[2d np array]
        Coordinates of points whose nearest neighbors are needed.
    x : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space.
    nn : int, optional
        Number of nearest neighbors. The default is 1.
    nmax : int, optional
        Maximum number of nearest neighbors. The default is 10.
    n_jobs : int, optional
        Number of processors to use. The default is -1 (all processors).
        
    Returns
    -------
    dist_nn : list[list]
        Distance of nearest neighbors.
    ind_nn : list[list]
        Index of nearest neighbors.

    """
        
    if isinstance(x_query, list):
        x_query = np.vstack(x_query)
    
    neigh = NearestNeighbors(n_neighbors=nn,
                             algorithm='auto',
                             metric='minkowski',
                             p=2,
                             n_jobs=-1)
    
    #Fit neighbors estimator object
    neigh.fit(X)
    
    #Ask for nearest neighbors
    dist_nn, ind_nn = neigh.kneighbors(x_query, nn+nmax, return_distance=True)
    
    #take only nonzero distance neighbors
    first_n = (dist_nn!=0).argmax(axis=1)
    last_n = first_n+nn
    
    ind_nn = [ind_nn[i,first_n[i]:last_n[i]] for i in range(len(first_n))]
    dist_nn = [dist_nn[i,first_n[i]:last_n[i]] for i in range(len(first_n))]
    
    return dist_nn, ind_nn


def valid_flows(t_sample, ts, tt=None, T=1):
    """
    Check which time intervals correspond to valid trajectories.

    Parameters
    ----------
    t_sample : list[int]
        Time indices corresponding to the sampled short trajectories.
    ts : list[int]
        Start of trajectory.
    tt : list[int], optional
        End of trajectory. The default is None.
    T : int or list[int], optional
        Time horizons. The default is 1.

    Returns
    -------
    tt : list[int]
        Start of trajectory.
    ts : list[int]
        End of trajectory.

    """
    
    r,c = ts.shape
    ts = ts.flatten()
    
    if isinstance(T, int) or np.issubdtype(T, np.integer):
        T *= np.ones_like(ts)
        
    if tt is None:
        tt = [ts[i]+T[i] for i in range(len(ts))]
    else:
        assert len(tt)==len(ts), 'Number of source points must equal to the \
            number of target points.'
            
    t_breaks = np.zeros_like(t_sample)
    t_breaks[np.array(t_sample)==0] = 1
    t_breaks[-max(T):] = 1
    
    ok = np.ones_like(tt)
    for i,t in enumerate(zip(ts,tt)):
        ok[i] = np.sum(t_breaks[t[0]:t[1]])==0
        
    tt = [tt[i] if ok[i]==1 else None for i in range(len(ok))]
    ts = [ts[i] if ok[i]==1 else None for i in range(len(ok))]
    
    ts = np.array(ts).reshape(r,c)
    tt = np.array(tt).reshape(r,c)
        
    return ts, tt


def generate_flow(X, ts, tt=None, T=10):
    """
    Obtain trajectories of between timepoints.

    Parameters
    ----------
    X : np array
        Trajectories.
    ts : int or np array or list[int]
        Source timepoint.
    tt : int or list[int]
        Target timepoint. The default is None.
    T : int
        Length of trajectory. Used when tt=None. The default is 10.

    Returns
    -------
    X_sample : list[np array].
        Set of flows of length T.

    """
    
    if not isinstance(ts, (list, tuple, np.ndarray)):
        ts = [ts]
    
    if tt is None:
        tt = [t+T if t is not None else None for t in ts]
    else:
        assert len(tt)==len(ts), 'Number of source points must equal to the \
            number of target points.'
    
    X_sample = []
    for s,t in zip(ts,tt):
        if s is None or t is None:
            X_sample.append(None)
        else:
            X_sample.append(X[s:t+1,:])
        
    return X_sample
    

def all_geodesic_dist(X, ts, tt, interp=False):
    """
    Compute all geodesic distances 

    Parameters
    ----------
    X : np array
        Datapoints.
    tt : list[int]
        Start of trajectory.
    ts : list[int]
        End of trajectory.
    interp : bool, optional
        Cubic interpolation between points. The default is false.

    Returns
    -------
    dst : np.array
        Geodesic distance from a set of timepoints with horizon T.

    """
    
    r,c = ts.shape
    ts = ts.flatten()
    tt = tt.flatten()
            
    dst = []
    for s,t in zip(ts,tt):
        if s is None or t is None:
            dst.append(None)
        else:
            dst.append(geodesic_dist(s, t, X, interp=interp))
        
    dst = np.array(dst).reshape(r,c)
    
    return dst


def geodesic_dist(s, t, x, interp=False):
    """
    Find the geodesic distance between points x1, x2

    Parameters
    ----------
    s : int
        Index of first endpoint of geodesic.
    t : int
        Index of second endpoint of geodesic.
    x : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space.
    interp : bool, optional
        Interpolate between points. The default is 0.

    Returns
    -------
    dist : float
        Geodesic distance.

    """
    
    assert s<t, 'First point must be before second point!'
    
    if interp:
        #compute spline through points
        tck, u = fit_spline(x.T, degree=3, smoothing=0.0, per_bc=0)
        u_int = [u[s], u[t]]
        x = eval_spline(tck, u_int, n=1000)
    else:
        x = x[s:t,:]
    
    dij = np.diff(x, axis=0)
    dij *= dij
    dij = dij.sum(1)
    dij = np.sqrt(dij)
        
    dist = dij.sum()
        
    return dist
    

def fit_spline(X, degree=3, smoothing=0.0, per_bc=0):
    """
    Fit spline to points

    Parameters
    ----------
    X : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space.
    degree : int, optional
        Order of spline. The default is 3.
    smoothing : float, optional
        Smoothing. The default is 0.0.
    per_bc : bool, optional
        Periodic boundary conditions (for closed curve). The default is 0.

    Returns
    -------
    tck : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.

    """
    
    tck, u = splprep(X, u=None, s=smoothing, per=per_bc, k=degree) 
    
    return tck, u


def eval_spline(tck, u_int, n=100):
    """
    Evaluate points on spline

    Parameters
    ----------
    tck : tuple (t,c,k)
        Vector of knots returned by splprep().
    u_int : list
        Parameter interval to evaluate the spline.
    n : int, optional
        Number of points to evaluate. The default is 100.

    Returns
    -------
    x_spline : TYPE
        DESCRIPTION.

    """
    
    u = np.linspace(u_int[0], u_int[1], n)
    x_spline = splev(u, tck, der=0)
    x_spline = np.vstack(x_spline).T
    
    return x_spline


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


def curvature_geodesic(dst):
    """
    Compute manifold curvature at a given set of points.

    Parameters
    ----------
    dst : np.array
        Geodesic distances for all points in the manifold (with a given time 
                                                           horizon, T).
        First row corresponds to geodesic distance on the manifold from a 
        set of points x(t) to points x(t+T).
        Rows from 2 to n correspond to geodesic distances between x(nn_t) and
        x(nn_t(t)+T), where nn_i(t) is the index of the nearest neighbor of 
        x_i(t) on attractor i.

    Returns
    -------
    kappa : list[float]
        List of curvature at timepoints t.

    """
    
    dst[dst==None] = np.nan
    dst[np.all(np.isnan(dst.astype(float)), axis=1)] = np.inf

    kappa = 1-np.nanmean(dst[:,1:],axis=1)/dst[:,0]
    kappa = kappa.astype(float)
     
    return kappa


def curvature_ball(X, ts, tt):

    n = ts.shape[1]
    kappa = np.zeros(n)
    # avg_vol = np.zeros(n)
    for i in range(ts.shape[1]):
        s = [i for i in ts[:,i] if i is not None]
        t = [i for i in tt[:,i] if i is not None]
        kappa[i] = 1- volume_simplex(X, t)/volume_simplex(X, s) 
        # avg_vol[i] = 0.5*(volume_simplex(X, s) + volume_simplex(X, t))
    
    # kappa = diff_vol/avg_vol
        
    return kappa


def volume_simplex(X,t):
    """
    Volume of convex hull of points

    Parameters
    ----------
    X : np.array
        Points on manifold.
    t : list[int]
        Time index of simplex vertices.

    Returns
    -------
    V : float
        Volume of simplex.

    """
    
    X_vertex = X[t,:]
    ch = ConvexHull(X_vertex)
    
    return ch.volume


def stack(X):
    """
    Stak ensemble of trajectories into attractor

    Parameters
    ----------
    X : list[np.array)]
        Individual trajectories in separate lists.

    Returns
    -------
    X_stacked : np.array
        Trajectories stacked.

    """
    
    X_stacked = np.vstack(X)
    
    return X_stacked


def unstack(X, t_sample):
    """
    Unstack attractor into ensemble of individual trajectories.

    Parameters
    ----------
    X : np.array
        Attractor.
    t_sample : list[list]
        Time indices of the individual trajectories.

    Returns
    -------
    X_unstack : list[np.array]
        Ensemble of trajectories.

    """
    
    X_unstack = []
    for t in t_sample:
        X_unstack.append(X[t,:])
        
    return X_unstack



# def embed_MDS(Y, dim=2):
#     """
#     Embed the feature distances into a lower dimensional space.

#     Parameters
#     ----------
#     Y : nxn matrix of feature distances
#         DESCRIPTION.
#     dim : int, optional
#         Dimension of embedding space. The default is 2.

#     Returns
#     -------
#     X_trnsf : TYPE
#         DESCRIPTION.

#     """
    
#     from sklearn.manifold import MDS
    
#     embedding = MDS(n_components=dim)
    
#     X_trnsf = embedding.fit_transform(Y)
    
#     return X_trnsf