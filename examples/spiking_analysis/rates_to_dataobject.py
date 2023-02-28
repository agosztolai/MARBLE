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

def main(separate_sessions=True, stop_crit=0.03, k=20, pca_n=5, gocue=10):        
    
    # instantaneous rate data
    rates =  pickle.load(open('../outputs/spiking_data/rate_data_50ms.pkl','rb'))       

    print('Converting to vector fields...')
    rates = start_at_gocue(rates, t = gocue)
    pca = fit_pca(rates, pca_n = pca_n)
    pos, vel = compute_velocity(rates, pca)
    pos, vel = remove_outliers(pos, vel)
    
    days, conditions = list(rates.keys()), list(rates[0].keys())
    
    with open('../outputs/spiking_data/data_pos_vel_50ms.pkl', 'wb') as handle:
        pickle.dump([pos, vel], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # print('Constructing data objects')
    # if separate_sessions:
    #     for i in range(len(days)):
    #         pos_, vel_ = [], []
    #         for j in range(len(conditions)):
    #             pos_.append(pos[j][i])
    #             vel_.append(vel[j][i])
                
    #         data = utils.construct_dataset(pos_, features=vel_, graph_type='cknn', k=k, stop_crit=stop_crit, n_workers=1,
    #                                    n_geodesic_nb=10, vector=False)
            
    #         with open('../outputs/spiking_data/data_dataobject_session_{}.pkl'.format(i), 'wb') as handle:
    #             pickle.dump([data, conditions], handle, protocol=pickle.HIGHEST_PROTOCOL)
    # else:     
    #     pos = [p for p_c in pos for p in p_c]
    #     vel = [v for v_c in vel for v in v_c]
    #     labels = [(c, d) for c in conditions for d in days]
    
    #     data = utils.construct_dataset(pos, features=vel, graph_type='cknn', k=k, stop_crit=stop_crit, n_workers=1,
    #                                    n_geodesic_nb=10, vector=False)

    #     with open('../outputs/spiking_data/data_dataobject_k{}.pkl'.format(k), 'wb') as handle:
    #         pickle.dump([data, labels], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print('Done')
        
        
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
            _pos, _vel = [], []
            for t in trials:
                p = pca.transform(t.T)
                v = np.diff(p, axis=0)
                p = p[:-1,:]
                
                _pos.append(p)
                _vel.append(v)
                
            pos[i].append(np.vstack(_pos))
            vel[i].append(np.vstack(_vel))
    
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

