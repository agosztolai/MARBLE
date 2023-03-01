
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

    # definingf the set of conditions     
    conditions=['DownLeft','Left','UpLeft','Up','UpRight','Right','DownRight']    
    
    # list of days
    days = rates.keys()
    
    # define some parameters
    pca_n = 5
    rm_outliers = False
    filter_data = False
    plot = False
      
    # storing all distance matrices
    embeddings = []
    distance_matrices = []
    

    times = [];
    all_condition_labels = [];
    
    # loop over each day
    for day in days:
        
        # first stack all trials from that day together and fit pca
        print(day)
        pos = []            
        # loop over each condition on that day
        for c, cond in enumerate(conditions):

            # go cue at 500ms 
            data = rates[day][cond][:,500:,:]
            
            # loop over all trials
            for trial in data:
                # store each trial as time x channels
                pos.append(trial)
                
                # smooth trial over time
                if filter_data:
                    trial = savgol_filter(trial, 9,2)
                
                # store each trial as time x channels
                pos.append(trial)
              
        # stacking all trials into a single array (time x channels)
        pos = np.vstack(pos)
        
        # fit PCA to all data across all conditions on a given day simultaneously
        pca = PCA(n_components=pca_n)
        pca.fit(pos)     
        print(pca.explained_variance_ratio_)   
        
        # create empty list of lists for each condition
        pos = [[] for u in range(len(conditions))]
        vel = [[] for u in range(len(conditions))]             
        timepoints = [[] for u in range(len(conditions))]    
        condition_labels = [[] for u in range(len(conditions))]    
        
        # loop over conditions
        for c, cond in enumerate(conditions):
            
            # go cue at 500ms 
            data = rates[day][cond][:,np.arange(500,1200,5),:]           
                       
            # loop over all trials
            for trial in data:
                
                # smooth trial over time
                if filter_data:
                    trial = savgol_filter(trial, 9,2)  
                
                # apply transformation to single trial
                trial = pca.transform(trial)
                
                # take all points except last
                pos[c].append(trial[:-1,:])
                
                # extract vectors between coordinates
                vel[c].append(get_vector_array(trial))
                
                timepoints[c].append(np.linspace(0,trial.shape[0]-2,trial.shape[0]-1))
                condition_labels[c].append(np.repeat(c,trial.shape[0]-1))
   
        # plt.quiver(pos[0][1][:,0],pos[0][1][:,1],vel[0][1][:,0],vel[0][1][:,1], angles='xy')
        
        # stack the trials within each condition
        pos = [np.vstack(u) for u in pos] # trials x time x channels
        vel = [np.vstack(u) for u in vel] # trials x time x channels
        timepoints = [np.hstack(u) for u in timepoints] 
        condition_labels = [np.hstack(u) for u in condition_labels]   
          
        # remove outliers
        if rm_outliers:
            pos, vel, timepoints, condition_labels = remove_outliers(pos, vel, timepoints, condition_labels)

        # construct data for marble

        data = utils.construct_dataset(pos, features=vel, graph_type='cknn', k=30, stop_crit=0.02, delta=1,
                                       n_nodes=None, n_workers=1, n_geodesic_nb=10, compute_laplacian=True, vector=False)

        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

        #x = data.x[:100,:].numpy()
        #pos = data.pos[:100,:].numpy()
        #plt.quiver(pos[:,0],pos[:,1],x[:,0],x[:,1], angles='xy')

        
        par = {'epochs': 150, #optimisation epochs
               'order': 2, #order of derivatives
               'hidden_channels': 100, #number of internal dimensions in MLP
               'out_channels': 6,
               'inner_product_features': False,
               'vec_norm': False,
               'diffusion': True,
              }
        
        model = net(data, **par)
        
        model.run_training(data, use_best=True, outdir='../../outputs/spiking_data/session_{}'.format(day))        
        data = model.evaluate(data)   
        
        # n_clusters = 50        
        # data = postprocessing(data, n_clusters=n_clusters)      
        

        # # extracting the time points associated with data
        # for c,cond in enumerate(conditions):
        #     time = timepoints[c]
        #     time = time[data.sample_ind[data.y==c]]
        #     timepoints[c] = time

        # embeddings.append(data.out)
        # distance_matrices.append(data.dist)
        # times.append(timepoints)
        # all_condition_labels.append(data.y)
        
        
        with open('./outputs/distance_matrices_and_embeddings_1ms_test.pkl', 'wb') as handle:
            pickle.dump([distance_matrices, embeddings, times , all_condition_labels ], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    if plot:
        emb_MDS, _ = geometry.embed(np.dstack(distance_matrices).mean(2), embed_typ = 'MDS')
        plt.figure()
        plt.scatter(emb_MDS[:,0],emb_MDS[:,1],c=np.linspace(0,6,7))
        plt.savefig('./outputs/mds_embedding.png')

        plt.figure()
        plt.imshow(np.dstack(distance_matrices).mean(2)); plt.colorbar()  
        plt.savefig('./outputs/distance_matrix.png')



                    
    
    
    return

def analyse():
    
    with open('./outputs/distance_matrices.pkl', 'rb') as handle:
        data = pickle.load(handle)

    distance_matrices = data[0]
    embeddings = data[1]
    train_distance_matrices = data[2]
    train_embeddings = data[3]
    
    # plot average distance matrix
    plt.figure()
    plt.imshow(np.median(np.dstack(train_distance_matrices),2)); plt.colorbar()  
    
    # plot average distance matrix
    emb_MDS, _ = geometry.embed(np.median(np.dstack(distance_matrices),2), embed_typ = 'MDS')
    plt.figure()
    plt.scatter(emb_MDS[:,0],emb_MDS[:,1],c=np.linspace(0,6,7))
    
    emb_MDS, _ = geometry.embed(np.median(np.dstack(train_distance_matrices),2), embed_typ = 'MDS')
    plt.figure()
    plt.scatter(emb_MDS[:,0],emb_MDS[:,1],c=np.linspace(0,6,7))
    
    

    #plt.savefig('./outputs/distance_matrix.png')


def get_vector_array(coords):
    """ function for defining the vector features from each array of coordinates """
    diff = np.diff(coords, axis=0)
    return diff

def remove_outliers(pos, vel, timepoints, condition_labels):
    """  function for removing outliers """
    clf = LocalOutlierFactor(n_neighbors=10)        
    # remove positional outliers
    for i,v in enumerate(pos):
        outliers = clf.fit_predict(v)
        vel[i] = vel[i][outliers==1]
        pos[i] = pos[i][outliers==1]  
        timepoints[i] = timepoints[i][outliers==1]
        condition_labels[i] = condition_labels[i][outliers==1]  
                         
    # remove velocity outliers
    for i,v in enumerate(vel):
        outliers = clf.fit_predict(v)
        vel[i] = vel[i][outliers==1]
        pos[i] = pos[i][outliers==1]  
        timepoints[i] = timepoints[i][outliers==1]
        condition_labels[i] = condition_labels[i][outliers==1]  
        
    return pos, vel, timepoints, condition_labels

if __name__ == '__main__':
    sys.exit(main())


