
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
    rates =  pickle.load(open('../data/rate_data_20ms.pkl','rb'))       
    trial_ids =  pickle.load(open('../data/trial_ids.pkl','rb'))       

    # definingf the set of conditions     
    conditions=['DownLeft','Left','UpLeft','Up','UpRight','Right','DownRight']    
    
    # list of days
    days = rates.keys()
    
    # define some parameters
    pca_n = 5
    rm_outliers = False
    filter_data = True
    plot = False
      
    # storing all distance matrices
    embeddings = []
    distance_matrices = []
    times = []; # to store the time point of each node in the trajectory
    all_condition_labels = []; # to store the condition label for each node
    all_trial_ids = []; # trial ids for each node
    all_sampled_ids = []; # to store all the nodes sampled by marble 
    
    # loop over each day
    for day in days:
        
        # first stack all trials from that day together and fit pca
        print(day)
        pos = []            
        # loop over each condition on that day
        for c, cond in enumerate(conditions):

            # go cue at 500ms (500ms / 20ms bin = 250)
            # only take rates from bin 10 onwards
            data = rates[day][cond][:,:,25:]
            
            # loop over all trials
            for t in range(data.shape[0]):
                
                # extract trial
                trial = data[t,:,:]
                
                # smooth trial over time
                if filter_data:
                    trial = savgol_filter(trial, 9,2)
                
                # store each trial as time x channels
                pos.append(trial.T)
              
        # stacking all trials into a single array (time x channels)
        pos = np.vstack(pos)
        
        # fit PCA to all data across all conditions on a given day simultaneously
        pca = PCA(n_components=pca_n)
        pca.fit(pos)        
        
        # create empty list of lists for each condition
        pos = [[] for u in range(len(conditions))]
        vel = [[] for u in range(len(conditions))]             
        timepoints = [[] for u in range(len(conditions))]    
        condition_labels = [[] for u in range(len(conditions))]  
        trial_indexes = [[] for u in range(len(conditions))]
        
        # loop over conditions
        for c, cond in enumerate(conditions):
            
            # go cue at 500ms (500ms / 50ms bin = 10)
            # only take rates from bin 10 onwards
            data = rates[day][cond][:,:,25:]           
            
            # loop over all trials
            for t in range(data.shape[0]):
                
                # extract trial
                trial = data[t,:,:]
                
                # smooth trial over time
                if filter_data:
                    trial = savgol_filter(trial, 9,2)  
                
                # apply transformation to single trial
                trial = pca.transform(trial.T)
                
                # take all points except last
                pos[c].append(trial[:-1,:])
                
                # extract vectors between coordinates
                vel[c].append(get_vector_array(trial))
                
                timepoints[c].append(np.linspace(0,trial.shape[0]-2,trial.shape[0]-1))
                condition_labels[c].append(np.repeat(c,trial.shape[0]-1))
                
                # adding trial id info (to match with kinematics decoding later)
                trial_indexes[c].append(np.repeat(trial_ids[day][cond][t],trial.shape[0]-1))
                
   
        # plt.quiver(pos[0][1][:,0],pos[0][1][:,1],vel[0][1][:,0],vel[0][1][:,1], angles='xy')
        
        # stack the trials within each condition
        pos = [np.vstack(u) for u in pos] # trials x time x channels
        vel = [np.vstack(u) for u in vel] # trials x time x channels
        timepoints = [np.hstack(u) for u in timepoints] 
        condition_labels = [np.hstack(u) for u in condition_labels]   
        trial_indexes = [np.hstack(u) for u in trial_indexes]   

        # remove outliers
        if rm_outliers:
            pos, vel, timepoints, condition_labels, trial_indexes = remove_outliers(pos, vel, timepoints, condition_labels, trial_indexes)

        # construct data for marble
        data = utils.construct_dataset(pos, features=vel, graph_type='cknn', k=30, stop_crit=0.0, delta=2.0,
                                       n_nodes=None, n_workers=1, n_geodesic_nb=10, compute_laplacian=True, vector=False)
        
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        
        #x = data.x[:100,:].numpy()
        #pos = data.pos[:100,:].numpy()
        #plt.quiver(pos[:,0],pos[:,1],x[:,0],x[:,1], angles='xy')

        
        par = {'epochs': 5, #optimisation epochs
               'order': 2, #order of derivatives
               'hidden_channels': 100, #number of internal dimensions in MLP
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
        

        embeddings.append(data.out)
        distance_matrices.append(data.dist)
        times.append(np.hstack(timepoints))
        all_condition_labels.append(data.y)
        all_trial_ids.append(np.hstack(trial_indexes))
        all_sampled_ids.append(data.sample_ind)

        
        with open('./outputs/distance_matrices_and_embeddings_20ms_sc02.pkl', 'wb') as handle:
            pickle.dump([distance_matrices, embeddings, times , all_condition_labels, all_trial_ids, all_sampled_ids], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
    
    return




def get_vector_array(coords):
    """ function for defining the vector features from each array of coordinates """
    diff = np.diff(coords, axis=0)
    return diff

def remove_outliers(pos, vel, timepoints, condition_labels, trial_indexes):
    """  function for removing outliers """
    clf = LocalOutlierFactor(n_neighbors=10)        
    # remove positional outliers
    for i,v in enumerate(pos):
        outliers = clf.fit_predict(v)
        vel[i] = vel[i][outliers==1]
        pos[i] = pos[i][outliers==1]  
        timepoints[i] = timepoints[i][outliers==1]
        condition_labels[i] = condition_labels[i][outliers==1]  
        trial_indexes[i] = trial_indexes[i][outliers==1]  
        
    # remove velocity outliers
    for i,v in enumerate(vel):
        outliers = clf.fit_predict(v)
        vel[i] = vel[i][outliers==1]
        pos[i] = pos[i][outliers==1]  
        timepoints[i] = timepoints[i][outliers==1]
        condition_labels[i] = condition_labels[i][outliers==1]  
        trial_indexes[i] = trial_indexes[i][outliers==1]  

    return pos, vel, timepoints, condition_labels, trial_indexes

if __name__ == '__main__':
    sys.exit(main())


