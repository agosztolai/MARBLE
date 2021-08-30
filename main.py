import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import splprep, splev

"""
We have the following functions:
    1) delay embedding (scalar and multi-d time series)
    2) standardize data
    3) find nearest neighbours to a point
    4) geodesic distance between two points (with and without smoothing spline)
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
        Time asymmetric or symmetric embedking. The default is 'asy'..

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


def find_nn(x_query, X, nn, n_jobs=-1):
    """
    Find nearest neighbours of a point on the manifold

    Parameters
    ----------
    x_query : 2d array, list[list] or list[array]
        Coordinates of points whose nearest neighbours are needed.
    x : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space.
    nn : int
        Number of nearest neighbours.
    n_jobs : int
        Number of processors to use. The default is -1 (all processors).
        
    Returns
    -------
    ind_nn : list[list]
        Index of nearest neighbours.

    """
    
    assert len(np.array(x_query).shape)==2, 'Query points are not in correct format.'
    
    neigh = NearestNeighbors(n_neighbors=nn,
                             algorithm='auto',
                             metric='minkowski',
                             p=2,
                             n_jobs=-1)
    
    #Fit neighbours estimator object
    neigh.fit(X)
    
    #Ask for nearest neighbours
    ind_nn = neigh.kneighbors(x_query, nn, return_distance=False)
    
    return ind_nn


def geodesic_dist(ind1, ind2, x, interp=False):
    """
    Find the geodesic distance between points x1, x2

    Parameters
    ----------
    ind1 : int
        Index of first endpoint of geodesic.
    ind2 : int
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
    
    assert ind1<ind2, 'First point must be before second point!'
    
    if interp:
        #compute spline through points
        tck, u = fit_spline(x.T, degree=3, smoothing=0.0, per_bc=0)
        u_int = [u[ind1], u[ind2]]
        x = eval_spline(tck, u_int, n=1000)
    else:
        x = x[ind1:ind2,:]
    
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
    