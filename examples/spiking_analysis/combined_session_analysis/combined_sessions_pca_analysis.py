
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

from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, MDS

from sklearn.neighbors import LocalOutlierFactor

from MARBLE import utils, geometry, net, plotting, postprocessing, compare_attractors

#%%   

def main():        
    """ fitting model for each day """    

    # instantaneous rate data
    rates =  pickle.load(open('../rate_data.pkl','rb'))       

    # definingf the set of conditions     
    conditions=['DownLeft','Left','UpLeft','Up','UpRight','Right','DownRight']    
    
    # list of days
    days = rates.keys()
    
    # define some parameters
    rm_outliers = True
    filter_data = False
    pca_n = 4
    plot = True
    
    pos = []

    # loop over each day
    for day in days:
     
        # loop over conditions
        for c, cond in enumerate(conditions):
            
            # go cue at 500ms (500ms / 50ms bin = 10)
            # only take rates from bin 10 onwards
            data = rates[day][cond][:,:,10:]           
                       
            # loop over all trials
            for t in range(data.shape[0]):
                
                # extra trial
                trial = data[t,:,:]
                
                # smooth trial over time 
                if filter_data:
                    trial = savgol_filter(trial, 10,2)  
                
                # transform to time x channels
                trial = trial.T
                
                # take all points except last
                pos.append(trial)
                
    # fit pca
    pca = PCA(n_components=pca_n)
    pca.fit(np.vstack(pos))    
    
    # create empty list of lists for each condition
    pos = [[] for u in range(len(conditions))]
    vel = [[] for u in range(len(conditions))]      
      
    # loop over each daydef analyse():
        
        with open('./outputs/all_data.pkl', 'rb') as handle:
            all_data = pickle.load(handle)
            
        distance_matrices = []
        for data in all_data:
            distance_matrices.append(data[0].dist)
            
        emb_MDS, _ = geometry.embed(np.dstack(distance_matrices).mean(2), embed_typ = 'MDS')
        plt.figure()
        plt.scatter(emb_MDS[:,0],emb_MDS[:,1],c=np.tile(np.linspace(0,6,7),[4,1]).flatten())
        
        plt.figure()
        plt.imshow(np.dstack(distance_matrices).mean(2)); plt.colorbar()  
        
    for day in days:
     
        # loop over conditions
        for c, cond in enumerate(conditions):
            
            # go cue at 500ms (500ms / 50ms bin = 10)
            # only take rates from bin 10 onwards
            data = rates[day][cond][:,:,10:]           
                       
            # loop over all trials
            for t in range(data.shape[0]):
                
                # extra trial
                trial = data[t,:,:]
                
                # smooth trial over time 
                if filter_data:
                    trial = savgol_filter(trial, 10,2)  
                
                # transform to time x channels
                trial = trial.T
                
                # embed in fitted pca space
                trial = pca.transform(trial)
                
                # take all points except last
                pos[c].append(trial[:-1,:])
                
                # extract vectors between coordinates
                vel[c].append(get_vector_array(trial))
                
   
    # plt.quiver(pos[0][1][:,0],pos[0][1][:,1],vel[0][1][:,0],vel[0][1][:,1], angles='xy')
        
    # stack the trials within each condition
    pos = [np.vstack(u) for u in pos] # trials x time x channels
    vel = [np.vstack(u) for u in vel] # trials x time x channels
    
   
    # remove outliers
    if rm_outliers:
        pos, vel = remove_outliers(pos, vel)

    # construct data for marble
    data = utils.construct_dataset(pos, features=vel, graph_type='cknn', k=40, stop_crit=0.05, 
                                   n_nodes=1000, n_workers=1, n_geodesic_nb=10, compute_cl=True, vector=False)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    
    par = {'epochs': 100, #optimisation epochs
           'order': 2, #order of derivatives
           'hidden_channels': 60, #number of internal dimensions in MLP
           'out_channels': 3,
           'inner_product_features': False,
           'vec_norm': False,
           'diffusion': True,
          }
    
    model = net(data, **par)
    
    model.run_training(data, use_best=True)        
    data = model.evaluate(data)        
    
    n_clusters = 50        
    data = postprocessing(data, n_clusters=n_clusters)
    
    if plot:
        emb_MDS, _ = geometry.embed(data.dist, embed_typ = 'MDS')
        plt.figure()
        plt.scatter(emb_MDS[:,0],emb_MDS[:,1],c=np.linspace(0,6,7))
        plt.savefig('./outputs/mds_embedding.png')

        plt.figure()
        plt.imshow(data.dist); plt.colorbar()  
        plt.savefig('./outputs/distance_matrix.png')

        plt.figure()
        plotting.embedding(data, data.y.numpy())
        plt.savefig('./outputs/node_embedding.png')


    with open('./outputs/data.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
            

    return

# def analyse():
#     import matplotlib.pyplot as plt
#     from MARBLE import geometry
#     import numpy as np
#     import pickle
    
#     with open('./outputs/data.pkl', 'rb') as handle:
#         data = pickle.load(open('./outputs/data.pkl', 'rb'))    

    
#     emb_MDS, _ = geometry.embed(data.dist, embed_typ = 'MDS')
#     plt.figure()
#     plt.scatter(emb_MDS[:,0],emb_MDS[:,1],c=np.linspace(0,6,7))
#     plt.savefig('./outputs/embedding.png')
    
    
#     plt.figure()
#     plt.imshow(data.dist); plt.colorbar()  
#     plt.savefig('./outputs/distance_matrix.png')
    


def get_vector_array(coords):
    """ function for defining the vector features from each array of coordinates """
    diff = np.diff(coords, axis=0)
    return diff

def remove_outliers(pos, vel):
    """  function for removing outliers """
    clf = LocalOutlierFactor(n_neighbors=10)        
    # remove positional outliers
    for i,v in enumerate(pos):
        outliers = clf.fit_predict(v)
        vel[i] = vel[i][outliers==1]
        pos[i] = pos[i][outliers==1]            
    # remove velocity outliers
    for i,v in enumerate(vel):
        outliers = clf.fit_predict(v)
        vel[i] = vel[i][outliers==1]
        pos[i] = pos[i][outliers==1]         
    return pos, vel

if __name__ == '__main__':
    sys.exit(main())


