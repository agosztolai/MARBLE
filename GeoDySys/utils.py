#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
from functools import partial
import torch

import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from cknn import cknneighbors_graph
from sklearn.model_selection import train_test_split



def construct_data_object(x, y, graph_type='knn', k=10):
        
    data_list = []
    G = []
    for i, y_ in enumerate(y):
        #fit knn before adding function as node attribute
        data_ = fit_knn_graph(x[i], k=k)
        data_.pos = torch.tensor(x[i])
            
        #save graph for testing and plotting
        G_ = to_networkx(data_, node_attrs=['x'], edge_attrs=None, to_undirected=True,
                remove_self_loops=True)
            
        #build pytorch geometric object
        data_.x = y_ #only function value as feature
        data_.num_nodes = len(x[i])
        data_.num_node_features = data_.x.shape[1]
        data_.y = torch.ones(data_.num_nodes, dtype=int)*i
        data_ = split(data_, test_size=0.1, val_size=0.5, seed=0)
        
        G.append(G_)
        data_list.append(data_)
        
    #collate datasets
    data_train, slices, _ = collate(data_list[0].__class__,
                                    data_list=data_list,
                                    increment=True,
                                    add_batch=False)
    
    return data_train, slices, G
    
    
def split(data, test_size=0.1, val_size=0.5, seed=0):
    """
    

    Parameters
    ----------
    data : pytorch geometric data object. All entried must be torch tensors
            for this to work properly.
    test_size : float between 0 and 1, optional
        Test set as fraction of total dataset. The default is 0.1.
    val_size : float between 0 and 1, optional
        Validation set as fraction of test set. The default is 0.5.
    seed : int, optional
        Seed. The default is 0.

    Returns
    -------
    data : pytorch geometric data object.

    """
    
    n = len(data.x)
    train_id, test_id = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    test_id, val_id = train_test_split(test_id, test_size=val_size, random_state=seed)
    
    train_mask = torch.zeros(n, dtype=bool)
    test_mask = torch.zeros(n, dtype=bool)
    val_mask = torch.zeros(n, dtype=bool)
    train_mask[train_id] = True
    test_mask[test_id] = True
    val_mask[val_id] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    
    return data


def fit_knn_graph(X, k=10, graph_type='cknn'):
    
    node_feature = torch.tensor(X, dtype=torch.float)
    
    ckng = cknneighbors_graph(node_feature, n_neighbors=k, delta=1.0)    
    if graph_type=='cknn':
        edge_index = torch.tensor(list(nx.Graph(ckng).edges), dtype=torch.int64).T
    elif graph_type=='knn':
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
