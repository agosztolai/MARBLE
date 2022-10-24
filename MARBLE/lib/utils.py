#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from torch import Tensor

import yaml
import os
from pathlib import Path
import warnings

from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, Batch
from torch_sparse import SparseTensor

import multiprocessing
from functools import partial
from tqdm import tqdm

from . import geometry

# =============================================================================
# Manage parameters
# =============================================================================
def parse_parameters(data, kwargs):
    """Load default parameters and merge with user specified parameters"""
    
    file = os.path.dirname(__file__) + '/../default_params.yaml'
    par = yaml.load(open(file,'rb'), Loader=yaml.FullLoader)
    
    #merge dictionaries without duplications
    for key in par.keys():
        if key not in kwargs.keys():
            kwargs[key] = par[key]
            
    kwargs['signal_dim'] = data.x.shape[1]
    kwargs['emb_dim'] = data.pos.shape[1]
    kwargs['slices'] = data._slice_dict['x']
    kwargs['n_sampled_nb'] = int(data.degree*par['frac_sampled_nb'])
    kwargs['n_geodesic_nb'] = int(data.degree*par['frac_geodesic_nb'])
    
    if kwargs['batch_norm']:
        kwargs['batch_norm'] = 'batch_norm'
            
    par = check_parameters(kwargs, data)
                  
    return par


def check_parameters(par, data):
    """Check parameter validity"""
                      
    if par['vector'] and data.x.shape[1]==1:
        par['vector'] = False
        warnings.warn('Vector computation is requested but signal dimension is \
                     one! Setting vector=False')
                     
    if par['frac_geodesic_nb'] <= 1.0:
        par['frac_geodesic_nb'] = 1.0
        warnings.warn('We need least the nearest neighbours to define the \
                      tangent space!')
                      
    assert par['order'] > 0, "Derivative order must be at least 1!" 
                      
    return par


def print_settings(model):
    """Print parameters to screen"""
    
    print('---- Settings: \n')
    
    for x in model.par:
        print (x,':',model.par[x])
        
    print('\n')
    
    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('---- Number of features to pass to the MLP: ', model.mlp.in_channels)
    print('---- Total number of parameters: ', np)


# =============================================================================
# Parallel processing
# =============================================================================
def parallel_proc(fun, iterable, inputs, processes=-1, desc=""):
    """Distribute an iterable function between processes"""
    
    if processes==-1:
        processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=processes)
    fun = partial(fun, inputs)
    result = list(tqdm(pool.imap(fun, iterable), 
                            total=len(iterable), 
                            desc=desc))
    pool.close()
    pool.join()
        
    return result

# =============================================================================
# Conversions
# =============================================================================
def construct_dataset(pos, features=None, graph_type='cknn', k=10):
    """Construct PyG dataset from node positions and features"""
                
    pos = [torch.tensor(p).float() for p in to_list(pos)]
    
    if features is not None:
        features = [torch.tensor(x).float() for x in to_list(features)]
        num_node_features = features[0].shape[1]
    else:
        num_node_features = None
        
    data_list = []
    for i, p in enumerate(pos):
        #fit graph to point cloud
        edge_index, edge_weight = geometry.fit_graph(p, 
                                                     graph_type=graph_type, 
                                                     par=k
                                                     )
        n = len(p)  
        data_ = Data(pos=pos[i], #positions
                     x=features[i], #features
                     edge_index=edge_index,
                     edge_weight=edge_weight,
                     num_nodes = n,
                     num_node_features = num_node_features,
                     y = torch.ones(n, dtype=int)*i
                     )
        
        data_list.append(data_)
        
    #collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    
    #split into training/validation/test datasets
    split = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    
    return split(batch)


def to_SparseTensor(edge_index, size=None, value=None):
    """
    Adjacency matrix as torch_sparse tensor

    Parameters
    ----------
    edge_index : (2x|E|) Matrix of edge indices
    size : pair (rows,cols) giving the size of the matrix. 
        The default is the largest node of the edge_index.
    value : list of weights. The default is unit values.

    Returns
    -------
    adj : adjacency matrix in SparseTensor format

    """    
    if value is not None:
        value = value[edge_index[0], edge_index[1]]
    if size is None:
        size = (edge_index.max()+1, edge_index.max()+1)
            
    adj = SparseTensor(row=edge_index[0], 
                       col=edge_index[1], 
                       value=value,
                       sparse_sizes=(size[0], size[1]))
    
    return adj


def np2torch(x, dtype=None):
    """Convert numpy to torch"""
    if dtype is None:
        return torch.from_numpy(x).float()
    elif dtype=='double':
        return torch.tensor(x, dtype=torch.int64)
    else:
        NotImplementedError


def to_list(x):
    """Convert to list"""
    if not isinstance(x, list):
        x = [x]
        
    return x


def to_pandas(x, augment_time=True):
    """Convert numpy to pandas"""
    columns = [str(i) for i in range(x.shape[1])]
    
    if augment_time:
        xaug = np.hstack([np.arange(len(x))[:,None], x])
        df = pd.DataFrame(xaug, 
                          columns = ['Time'] + columns, 
                          index = np.arange(len(x)))
    else:
        df = pd.DataFrame(xaug, 
                          columns = columns, 
                          index = np.arange(len(x)))
        
    return df


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)
    

# =============================================================================
# Statistics
# =============================================================================
def standardise(X, zero_mean=True, norm='std'):
    """Standarsise data row-wise"""
    
    if zero_mean:
        X -= X.mean(axis=0, keepdims=True)
    
    if norm=='std':
        X /= X.std(axis=0, keepdims=True)
    elif norm=='max':
        X /= abs(X).max(axis=0, keepdims=True)
    else:
        NotImplementedError
        
    return X


# =============================================================================
# Input/output
# =============================================================================
def _savefig(fig, folder, filename, ext):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        if not Path(folder).exists():
            os.mkdir(folder)
        fig.savefig((Path(folder) / filename).with_suffix(ext), bbox_inches="tight")