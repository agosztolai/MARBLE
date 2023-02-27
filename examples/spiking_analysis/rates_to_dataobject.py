#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import pickle
import sys

# from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA#, KernelPCA
# from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, MDS

from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
from MARBLE import utils

#%%   

def main():        
    
    # instantaneous rate data
    rates =  pickle.load(open('../outputs/spiking_data/rate_data.pkl','rb'))       

    # definingf the set of conditions     
    conditions=['DownLeft','Left','UpLeft','Up','UpRight','Right','DownRight']    
    
    # list of days
    days = rates.keys()
    
    # define some parameters
    rm_outliers = True
    pca_n = 4
    
    pos = []

    # loop over each day
    for day in tqdm(days):
     
        # loop over conditions
        for c, cond in enumerate(conditions):
            
            # go cue at 500ms (500ms / 50ms bin = 10)
            # only take rates from bin 10 onwards
            data = rates[day][cond][:,:,10:]           
                       
            # loop over all trials
            for t in range(data.shape[0]):
                
                # extra trial
                trial = data[t,:,:] 
                
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
      
    # loop over each day
    for day in tqdm(days):
     
        # loop over conditions
        for c, cond in enumerate(conditions):
            
            # go cue at 500ms (500ms / 50ms bin = 10)
            # only take rates from bin 10 onwards
            data = rates[day][cond][:,:,10:]           
                       
            # loop over all trials
            for t in range(data.shape[0]):
                
                # extra trial
                trial = data[t,:,:]
                
                # transform to time x channels
                trial = trial.T
                
                # embed in fitted pca space
                trial = pca.transform(trial)
                
                # take all points except last
                pos[c].append(trial[:-1,:])
                
                # extract vectors between coordinates
                vel[c].append(get_vector_array(trial))
                        
    # stack the trials within each condition
    pos = [np.vstack(u) for u in pos] # trials x time x channels
    vel = [np.vstack(u) for u in vel] # trials x time x channels
    
   
    # remove outliers
    if rm_outliers:
        pos, vel = remove_outliers(pos, vel)

    # construct data for marble
    data = utils.construct_dataset(pos, features=vel, graph_type='cknn', k=40, stop_crit=0.05, 
                                   n_geodesic_nb=10, compute_cl=True, vector=False)
    

    with open('../outputs/spiking_data/dataobject.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        

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

