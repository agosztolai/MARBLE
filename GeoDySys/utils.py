#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
from functools import partial
import torch

import networkx as nx
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from cknn import cknneighbors_graph


def fit_knn_graph(X, t_sample, k=10, method='cknn'):
    
    node_feature = [list(X[i]) for i in t_sample]
    node_feature = torch.tensor(node_feature, dtype=torch.float)
    
    ckng = cknneighbors_graph(node_feature, n_neighbors=k, delta=1.0)    
    if method=='cknn':
        edge_index = torch.tensor(list(nx.Graph(ckng).edges), dtype=torch.int64).T
    elif method=='knn':
        edge_index = knn_graph(node_feature, k=k)
    else:
        NotImplementedError
    
    edge_index = to_undirected(edge_index)
    
    data = Data(x=node_feature, edge_index=edge_index)
    
    return data



def parallel_proc(fun, iterable, inputs, processes=-1, desc=""):
    if processes==-1:
        processes = multiprocessing.cpu_count()
    pool = Pool(processes=processes)
    fun = partial(fun, inputs)
    result = list(tqdm(pool.imap(fun, iterable), 
                            total=len(iterable), 
                            desc=desc)
                  )
    pool.close()
    pool.join()
        
    return result


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


def standardize(X, axis=0):
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
