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

from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, Batch
from torch_sparse import SparseTensor

import multiprocessing
from functools import partial
from tqdm import tqdm

from . import geometry
from MARBLE import preprocessing

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def construct_dataset(pos, 
                      features, 
                      graph_type='cknn', 
                      k=10,
                      n_geodesic_nb=10,
                      stop_crit=0.0,
                      number_of_resamples=1,
                      n_nodes=None,
                      compute_cl=False,
                      n_workers=1,
                      vector=True,
                      dim_man=None,
                      labels=None):
    
    """Construct PyG dataset from node positions and features"""
                
    pos = [torch.tensor(p).float() for p in to_list(pos)]
    
    if not labels:
        labels = np.linspace(0,len(pos)-1,len(pos))

    if features is not None:
        features = [torch.tensor(x).float() for x in to_list(features)]
        num_node_features = features[0].shape[1]
    else:
        num_node_features = None
        
    if stop_crit==0.0:
        number_of_resamples=1
        
    data_list = []
    for i, (p, f) in enumerate(zip(pos, features)):
        for j in range(number_of_resamples):
            #even sampling of points
            start_idx = torch.randint(low=0, high=len(p), size=(1,))
            sample_ind, _ = geometry.furthest_point_sampling(p, 
                                                             stop_crit=stop_crit,
                                                             N=n_nodes,
                                                             start_idx=start_idx)
            p, f = p[sample_ind], f[sample_ind]
            
            #fit graph to point cloud
            edge_index, edge_weight = geometry.fit_graph(p, 
                                                         graph_type=graph_type, 
                                                         par=k
                                                         )
            n = len(p)  
            data_ = Data(pos=p, #positions
                         x=f, #features
                         edge_index=edge_index,
                         edge_weight=edge_weight,
                         num_nodes = n,
                         num_node_features = num_node_features,
                         y = torch.ones(n, dtype=int)*labels[i]
                         )
        
            data_list.append(data_)
        
    #collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    batch.number_of_resamples=number_of_resamples
    
    #split into training/validation/test datasets
    split = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    split(batch)
    
    batch = preprocessing.preprocessing(batch, 
                                        vector=vector, 
                                        compute_cl=compute_cl,
                                        n_workers=n_workers,
                                        n_geodesic_nb=n_geodesic_nb,
                                        dim_man=dim_man)
    
    return batch


# =============================================================================
# Manage parameters
# =============================================================================
def parse_parameters(data, kwargs):
    """Load default parameters and merge with user specified parameters"""
    
    file = os.path.dirname(__file__) + '/../default_params.yaml'
    par = yaml.load(open(file,'rb'), Loader=yaml.FullLoader)
    
    par['dim_signal'] = data.x.shape[1]
    par['dim_emb'] = data.pos.shape[1]
    
    if hasattr(data, 'dim_man'):
        par['dim_man'] = data.dim_man
    
    #merge dictionaries without duplications
    for key in par.keys():
        if key not in kwargs.keys():
            kwargs[key] = par[key]
            
    if par['frac_sampled_nb']!=-1:
        kwargs['n_sampled_nb'] = int(data.degree*par['frac_sampled_nb'])
    else:
        kwargs['n_sampled_nb'] = -1
        
    if kwargs['batch_norm']:
        kwargs['batch_norm'] = 'batch_norm'
    else:
        kwargs['batch_norm'] = None
            
    par = check_parameters(kwargs, data)
                  
    return par


def check_parameters(par, data):
    """Check parameter validity"""
                      
    assert par['order'] > 0, "Derivative order must be at least 1!"
    
    if par['vec_norm']:
        assert data.x.shape[1] > 1, 'Using vec_norm=True is \
            not permitted for scalar signals'
        
    if par['diffusion']:
        assert hasattr(data, 'L'), 'No Laplacian found. Compute it in preprocessing()!'
        
    pars = ['batch_size', 'epochs', 'lr', 'momentum', 'order', \
            'inner_product_features', 'dim_signal', 'dim_emb', 'dim_man',\
            'frac_sampled_nb', 'dropout', 'n_lin_layers', 'diffusion', \
            'hidden_channels', 'out_channels', 'bias', 'batch_norm', 'vec_norm', \
            'seed', 'n_sampled_nb', 'processes']
        
    for p in par.keys():
        assert p in pars, 'Unknown specified parameter {}!'.format(p)
                      
    return par


def print_settings(model):
    """Print parameters to screen"""
    
    print('\n---- Settings: \n')
        
    for x in model.par:
        print (x,':',model.par[x])
            
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_features = model.enc.in_channels
    
    print('\n---- Number of features to pass to the MLP: ', n_features)
    print('---- Total number of parameters: ', n_parameters)


# =============================================================================
# Parallel processing
# =============================================================================
def parallel_proc(fun, iterable, inputs, processes=-1, desc=""):
    """Distribute an iterable function between processes"""
    
    if processes==-1:
        processes = multiprocessing.cpu_count()
        
    if processes>1 and len(iterable)>1:
        pool = multiprocessing.Pool(processes=processes)
        fun = partial(fun, inputs)
        result = list(tqdm(pool.imap(fun, iterable), 
                           total=len(iterable), 
                           desc=desc))
        pool.close()
        pool.join()
    else:
        result = [fun(inputs, i) for i in tqdm(iterable, desc=desc)]
        
    return result


def move_to_gpu(model, data, adjs=None):
    """Move stuff to gpu"""
    
    assert hasattr(data, 'kernels'), \
        'It seems that data is not preprocessed. Run preprocess(data)!'
    
    model = model.to(device)
    x = data.x.to(device)
    
    if hasattr(data, 'L'):
        L = [l.to(device) for l in data.L]
    else:
        L = None
        
    Lc = data.Lc.to(device) if hasattr(data, 'Lc') else None
    kernels = [K.to(device) for K in data.kernels]
    gauges = data.gauges.to(device)
            
    if adjs is None:
        return model, x, L, Lc, kernels, gauges
    else:
        adjs = [adj.to(device) for adj in adjs]
        return model, x, L, Lc, kernels, gauges, adjs


def detach_from_gpu(model, data, adjs=None):
    """detach stuff from gpu"""
    
    assert hasattr(data, 'kernels'), \
        'It seems that data is not preprocessed. Run preprocess(data)!'
    
    model = model.to(device)
    x = data.x.detach().cpu()
    
    if hasattr(data, 'L'):
        L = [l.detach().cpu() for l in data.L]
    else:
        L = None
        
    Lc = data.Lc.detach().cpu() if hasattr(data, 'Lc') else None
    kernels = [K.detach().cpu() for K in data.kernels]
    gauges = data.gauges.detach().cpu()
            
    if adjs is None:
        return model, x, L, Lc, kernels, gauges
    else:
        for i, adj in enumerate(adjs):
            adjs[i] = [adj[0].detach().cpu(), adj[1].detach().cpu(), adj[2]]
        return model, x, L, Lc, kernels, gauges, adjs
    

# =============================================================================
# Conversions
# =============================================================================
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
    if value is None:
        value = torch.ones(edge_index.shape[1])
    if size is None:
        size = (int(edge_index[0].max())+1, int(edge_index[1].max())+1)
        
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
    

def expand_index(ind, dim):
    """Interleave dim incremented copies of ind"""
    
    n = len(ind)
    ind = [ind*dim+i for i in range(dim)]
    ind = torch.hstack(ind).view(dim, n).t().flatten()
    
    return ind


def to_block_diag(sp_tensors):
    
    sizes = [torch.tensor(t.size()).unsqueeze(1) for t in sp_tensors]
    ind = [t.indices() for t in sp_tensors]
    val = [t.values() for t in sp_tensors]

    for i in range(1,len(sp_tensors)):
        for j in range(i):
            ind[i] += sizes[j]
        
    ind = torch.hstack(ind)
    val = torch.hstack(val)

    return torch.sparse_coo_tensor(ind, val)
    
        
def expand_edge_index(edge_index, dim=1):
    """When using rotations, we replace nodes by vector spaces so
       need to expand adjacency matrix from nxn -> n*dimxn*dim matrices"""
       
    dev = edge_index.device
    if dev!='cpu':
        edge_index = edge_index.to('cpu')
       
    n = edge_index.shape[1]
    ind = [torch.tensor([i,j]) for i in range(dim) for j in range(dim)]
    edge_index = [edge_index*dim+i.unsqueeze(1) for i in ind]
    edge_index = torch.stack(edge_index, dim=2).view(2,n*len(ind))
    
    if dev!='cpu':
        edge_index.to(dev)
        
    return edge_index


def tile_tensor(tensor, dim):
    """Enlarge nxn tensor to d*dim x n*dim block matrix. Effectively
    computing a sparse version of torch.kron(K, torch.ones((dim,dim)))"""
    
    tensor = tensor.coalesce()
    edge_index = tensor.indices()
    edge_index = expand_edge_index(edge_index, dim=dim)
    return torch.sparse_coo_tensor(edge_index, 
                                   tensor.values().repeat_interleave(dim*dim)) 


def restrict_dimension(sp_tensor, d, m):
    """Limit the dimension of the tensor"""
    n = sp_tensor.size(0)
    idx = torch.ones(n)
    for i in range(m,d):
        idx[m::d] = 0      
    idx = torch.where(idx)[0]
    sp_tensor = torch.index_select(sp_tensor, 0, idx).coalesce()
    return torch.index_select(sp_tensor, 1, idx).coalesce()
    

def restrict_to_batch(sp_tensor, idx):
    """Restrict tensor to current batch"""
    
    idx = [i.to(sp_tensor.device) for i in idx]
    
    if len(idx)==1:
        return torch.index_select(sp_tensor, 0, idx[0]).coalesce()
    elif len(idx)==2:
        sp_tensor = torch.index_select(sp_tensor, 0, idx[0])
        return torch.index_select(sp_tensor, 1, idx[1]).coalesce()

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