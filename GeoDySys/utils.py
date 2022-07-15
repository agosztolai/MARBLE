#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import yaml
import os
import warnings

import networkx as nx

from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor

from torch_geometric.nn import knn_graph, radius_graph
from cknn import cknneighbors_graph

import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def parse_parameters(model, data, kwargs):
    """Load default parameters and merge with user specified parameters"""
    
    file = os.path.dirname(__file__) + '/default_params.yaml'
    par = yaml.load(open(file,'rb'), Loader=yaml.FullLoader)
    
    #merge dictionaries without duplications
    for key in par.keys():
        if key not in kwargs.keys():
            kwargs[key] = par[key]
                  
    return kwargs


def check_parameters(par, data):
    if data.degree >= par['n_geodesic_nb']:
        par['n_geodesic_nb'] = data.degree
        warnings.warn('Number of geodesic neighbours (n_geodesic_nb) should \
                      be (ideally) greater than the number of neighbours!')
    
    if data.degree < par['n_sampled_nb']:
        par['n_sampled_nb'] = data.degree
        warnings.warn('Sampled points (n_nb_samples) exceeds the degree (k)\
                      of the graph! Continuing with n_nb_samples=k... ')
                      
    return par


def print_settings(model, out_channels):
    
    print('---- Settings: \n')
    
    for x in model.par:
        print (x,':',model.par[x])
        
    print('\n')
    
    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('---- Number of channels to pass to the MLP: ', out_channels)
    print('---- Total number of parameters: ', np)


def adjacency_matrix(edge_index, size=None, value=None):
    
    if value is not None:
        value=value[edge_index[0], edge_index[1]]
    if size is None:
        size = (edge_index.max()+1, edge_index.max()+1)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                       value=value,
                       sparse_sizes=(size[0], size[1]))
    
    return adj


def construct_dataset(positions, features=None, graph_type='cknn', k=10, connect_datasets=False):
    """Construct PyG dataset from node positions and features"""
        
    positions = tolist(positions)
    features = tolist(features)
        
    positions = [torch.tensor(p).float() for p in positions]
    features = [torch.tensor(x).float() for x in features]
        
    data_list = []
    for i, x in enumerate(features):
        edge_index = fit_graph(positions[i], graph_type=graph_type, par=k)
        n = len(positions[i])
        
        if connect_datasets:
            if i < len(features)-1:
                edge_index = torch.hstack([edge_index, 
                                           torch.tensor([n-1,n]).unsqueeze(-1)])
            
        data_ = Data(x=positions[i], edge_index=edge_index)
        
        if x is None:
            A = adjacency_matrix(edge_index)
            x = A.sum(1).unsqueeze(-1)
            
        data_.pos = positions[i] #positions
        data_.x = x #features
        data_.num_nodes = n
        data_.num_node_features = data_.x.shape[1]
        data_.y = torch.ones(data_.num_nodes, dtype=int)*i
        
        data_list.append(data_)
        
    #collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    
    #split into training/validation/test datasets
    split = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    
    return split(batch)


def fit_graph(x, graph_type='cknn', par=1):
    """Fit graph to node positions"""
    
    ckng = cknneighbors_graph(x, n_neighbors=par, delta=1.0)
    
    if graph_type=='cknn':
        edge_index = torch.tensor(list(nx.Graph(ckng).edges), dtype=torch.int64).T
    elif graph_type=='knn':
        edge_index = knn_graph(x, k=par)
    elif graph_type=='radius':
        edge_index = radius_graph(x, r=par)
    else:
        NotImplementedError
    
    edge_index = to_undirected(edge_index)
    
    return edge_index


def parallel_proc(fun, iterable, inputs, processes=-1, desc=""):
    """Parallel processing, distribute an iterable function between processes"""
    
    if processes==-1:
        processes = multiprocessing.cpu_count()
    pool = Pool(processes=processes)
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
def torch2np(x):
    return x.detach().to(torch.device('cpu')).numpy()


def np2torch(x):
    return torch.from_numpy(x).float()


def tolist(x):
    if not isinstance(x, (list,tuple)):
        x = [x]
        
    return x