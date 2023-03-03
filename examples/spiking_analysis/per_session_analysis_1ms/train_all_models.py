#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import pickle
import random
import os
import sys

import torch.nn as nn
import torch

import itertools

from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA, KernelPCA

from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, MDS

from sklearn.neighbors import LocalOutlierFactor

from MARBLE import utils, geometry, net, plotting, postprocessing, compare_attractors

#%%   

def main():    
    
    """ fitting model for each day + pca embedding """    
    
    # instantaneous rate data
    rates = pickle.load(open('../../outputs/spiking_data/rate_data_1ms.pkl','rb'))       
    
    # list of days
    days = rates.keys()
    
    
    # loop over each day
    for day in days:
        if day<22:
            continue
            
        # load data for marble
        data = pickle.load(open('../../outputs/spiking_data/data_object_session_{}.pkl'.format(day),'rb'))
        

        par = {'epochs': 150, #optimisation epochs
               'order': 2, #order of derivatives
               'hidden_channels': 32, #number of internal dimensions in MLP
               'out_channels': 8,
               'inner_product_features': False,
               'diffusion': True,
               }
        
        model = net(data, **par)
        
        model.run_training(data, use_best=True, outdir='../../outputs/spiking_data/session_{}'.format(day))        
        data = model.evaluate(data)   
        data = postprocessing(data, n_clusters=50)
        
        with open('../../outputs/spiking_data/session_{}/data_object_session_{}.pkl'.format(day,day), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



# def analyse():
    
#     with open('./outputs/distance_matrices.pkl', 'rb') as handle:
#         data = pickle.load(handle)

#     distance_matrices = data[0]
#     embeddings = data[1]
#     train_distance_matrices = data[2]
#     train_embeddings = data[3]
    
#     # plot average distance matrix
#     plt.figure()
#     plt.imshow(np.median(np.dstack(train_distance_matrices),2)); plt.colorbar()  
    
#     # plot average distance matrix
#     emb_MDS, _ = geometry.embed(np.median(np.dstack(distance_matrices),2), embed_typ = 'MDS')
#     plt.figure()
#     plt.scatter(emb_MDS[:,0],emb_MDS[:,1],c=np.linspace(0,6,7))
    
#     emb_MDS, _ = geometry.embed(np.median(np.dstack(train_distance_matrices),2), embed_typ = 'MDS')
#     plt.figure()
#     plt.scatter(emb_MDS[:,0],emb_MDS[:,1],c=np.linspace(0,6,7))
    
    

#     #plt.savefig('./outputs/distance_matrix.png')


def get_vector_array(coords):
    """ function for defining the vector features from each array of coordinates """
    diff = np.diff(coords, axis=0)
    return diff

if __name__ == '__main__':
    sys.exit(main())


