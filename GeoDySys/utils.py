#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import yaml
import os
from pathlib import Path
import warnings

from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, Batch

import multiprocessing
from functools import partial
from tqdm import tqdm

from GeoDySys import geometry


# =============================================================================
# Manage parameters
# =============================================================================
def parse_parameters(data, kwargs):
    """Load default parameters and merge with user specified parameters"""
    
    file = os.path.dirname(__file__) + '/default_params.yaml'
    par = yaml.load(open(file,'rb'), Loader=yaml.FullLoader)
    
    #merge dictionaries without duplications
    for key in par.keys():
        if key not in kwargs.keys():
            kwargs[key] = par[key]
            
    kwargs['signal_dim'] = data.x.shape[1]
    kwargs['emb_dim'] = data.pos.shape[1]
            
    par = check_parameters(kwargs, data)
                  
    return par


def check_parameters(par, data):
    """Check parameter validity"""
    if data.degree >= par['n_geodesic_nb']:
        par['n_geodesic_nb'] = data.degree
        warnings.warn('Number of geodesic neighbours (n_geodesic_nb) should \
                      be (ideally) greater than the number of neighbours!')
    
    if data.degree < par['n_sampled_nb']:
        par['n_sampled_nb'] = data.degree
        warnings.warn('Sampled points (n_nb_samples) exceeds the degree (k)\
                      of the graph! Continuing with n_nb_samples=k... ')
                      
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
        
    pos = tolist(pos)
    features = tolist(features)
        
    pos = [torch.tensor(p).float() for p in pos]
    features = [torch.tensor(x).float() for x in features]
        
    data_list = []
    for i, x in enumerate(features):
        #fit graph to point cloud
        edge_index, edge_weight = geometry.fit_graph(pos[i], 
                                                     graph_type=graph_type, 
                                                     par=k
                                                     )
        n = len(pos[i])  
        data_ = Data(pos=pos[i], #positions
                     x=x, #features
                     edge_index=edge_index,
                     edge_weight=edge_weight,
                     num_nodes = n,
                     num_node_features = x.shape[1],
                     y = torch.ones(n, dtype=int)*i
                     )
        
        data_list.append(data_)
        
    #collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    
    #split into training/validation/test datasets
    split = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    
    return split(batch)


def torch2np(x):
    return x.detach().to(torch.device('cpu')).numpy()


def np2torch(x, dtype=None):
    if dtype is None:
        return torch.from_numpy(x).float()
    elif dtype=='double':
        return torch.tensor(x, dtype=torch.int64)
    else:
        NotImplementedError


def tolist(x):
    if not isinstance(x, (list,tuple)):
        x = [x]
        
    return x

# =============================================================================
# Input/output
# =============================================================================
def _savefig(fig, folder, filename, ext):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        if not Path(folder).exists():
            os.mkdir(folder)
        fig.savefig((Path(folder) / filename).with_suffix(ext), bbox_inches="tight")