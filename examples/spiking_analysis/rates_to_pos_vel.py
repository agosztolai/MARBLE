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

#%%   

def main():        
    
    # instantaneous rate data
    rates =  pickle.load(open('../outputs/spiking_data/rate_data.pkl','rb'))       

    rates = start_at_gocue(rates, t = 500)
    pca = fit_pca(rates, pca_n = 3)
    pos, vel = compute_velocity(rates, pca)
    pos, vel = remove_outliers(pos, vel)
    

    with open('../outputs/spiking_data/dataobject.pkl', 'wb') as handle:
        pickle.dump([pos, vel], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def start_at_gocue(rates, t = 500):
    
    days, conditions = rates.keys(), rates[0].keys()
    
    for d in days:
        for c in conditions:
            rates[d][c] = rates[d][c][:,:,t:]
        
    return rates
                       
        
def fit_pca(rates, pca_n = 3):
    
    days, conditions = rates.keys(), rates[0].keys()
    
    pos = []
    for d in days:
        for c in conditions:
            trials = rates[d][c]
            for t in trials:
                pos.append(t.T)
                
    # fit pca
    pca = PCA(n_components=pca_n)
    pca.fit(np.vstack(pos))
    print(pca.explained_variance_ratio_)
    
    return pca


def compute_velocity(rates, pca):
    
    days, conditions = rates.keys(), rates[0].keys()
    
    # create empty list of lists for each condition
    pos = [[] for u in range(len(conditions))]
    vel = [[] for u in range(len(conditions))]      
      
    # loop over each day
    for d in days:
     
        # loop over conditions
        for i, c in enumerate(conditions):
            
            trials = rates[d][c]   
                       
            # loop over all trials
            for t in trials:
                p = pca.transform(t.T)
                v = np.diff(p, axis=0)
                p = p[:-1,:]
                
                pos[i].append(p)
                vel[i].append(v)
    
    return pos, vel


def remove_outliers(pos, vel):
        
    for i, (p_day, v_day) in tqdm(enumerate(zip(pos, vel))): 
        for j, (p, v) in enumerate(zip(p_day, v_day)):
    
            p, v = _remove_outliers(p, v)
            pos[i][j], vel[i][j] = p, v
    
    return pos, vel


def _remove_outliers(pos, vel):
    """  function for removing outliers """
    clf = LocalOutlierFactor(n_neighbors=10)        
    # remove positional outliers
    outliers = clf.fit_predict(pos)
    vel = vel[outliers==1]
    pos = pos[outliers==1]            
    # remove velocity outliers
    outliers = clf.fit_predict(vel)
    vel = vel[outliers==1]
    pos = pos[outliers==1]         
    return pos, vel


if __name__ == '__main__':
    sys.exit(main())

