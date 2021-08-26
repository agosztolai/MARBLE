import numpy as np


def delay_embed(x, k, tau=1, typ='asy'):
    """
    Delay-embedking for scalar time-series x(t).
    Builds prekictive coorkinates Y=[x(t), x(t-tau), ..., x(t-(k-1)*tau)]

    Parameters
    ----------
    x : Tx1 array
        Scalar time series.
    k : int
        Embedking kension. Needs to be at most 2*d+1, possibly smaller.
    tau : int, optional
        Delay parameter. The default is 1.
    typ : 'asy', 'sym', optional
        Time asymmetric or symmetric embedking. The default is 'asy'.

    Returns
    -------
    Y : kx(T-(k-1)*tau) numpy array
        Delay embedded coorkinates.

    """
    
    T = len(x)
    N = T - (k-1)*tau # length of embedded signal  
    Y = np.zeros([N,k])
        
    if typ == 'asy':  
        for ki in range(k):
            ind = np.arange(ki*tau, N+ki*tau)
            Y[:,ki] = x[ind]
            
    elif typ == 'sym' and k % 2 == 0:
        for ki in range(-k//2,k//2):
            ind = np.arange((k//2+ki)*tau, N+(k//2+ki)*tau)
            Y[:,ki] = x[ind]
            
    elif typ == 'sym' and k % 2 == 1:    
        for ki in range(-(k-1)//2,(k-1)//2+1):
            ind = np.arange(((k-1)//2+ki)*tau, N+((k-1)//2+ki)*tau)
            Y[:,ki] = x[ind]
          
    return Y


# def delay_embed_multid(Y,k,tau,typ='asy'):
    
#     if len(Y.shape) == 1:
#         return delay_embed(Y,k,tau,typ)
        
#     dtaumax = np.max((k-1)*tau)
#     nrow = Y.shape[0] - dtaumax # length of embedded signal
#     D = Y.shape[1] # number of variables
    
#     if type(tau) is int:
#         tau = np.repeat(tau,D)
        
#     if type(k) is int:
#         k = np.repeat(k,D)
        
#     X = np.zeros([nrow,np.sum(k)])
        
#     assert (len(tau) == D) & (len(k) == D)
    
#     if typ == 'asy':
#         for i in range(D):
#             for ki in range(k[i]):
#                 ind = np.sum(k[:i]) + ki
#                 X[:,ind] = Y[ np.arange(ki*tau[i], nrow+ki*tau[i]), i ]
            
#     elif typ == 'sym' and k[i] % 2 == 0:
#         for i in range(D):
#             for ki in range(-k[i]//2,k[i]//2):
#                 ind = np.sum(k[:i]) + k[i]//2 + ki 
#                 X[:,ind] = Y[ np.arange((k[i]//2+ki)*tau[i], nrow+(k[i]//2+ki)*tau[i]), i ]
            
#     elif typ == 'sym' and k[i] % 2 == 1:
#         for i in range(D):
#             for ki in range(-(k[i]-1)//2,(k[i]-1)//2+1):
#                 ind = np.sum(k[:i]) + (k[i]-1)//2 + ki
#                 X[:,ind] = Y[ np.arange(((k[i]-1)//2+ki)*tau[i], nrow+((k[i]-1)//2+ki)*tau[i]), i ]
    
#     return X