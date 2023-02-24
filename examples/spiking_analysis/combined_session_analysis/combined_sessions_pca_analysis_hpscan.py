
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
    rm_outliers = False
    filter_data = True
    plot = True
    save = False
    
    # parameters to scan
    n_nodes = [1000,2000,3000,4000]
    pca_n = [4]
    epochs = [100]
    hidden_channels = [60]
    order = [2]
    k_neighbours = [20,40]
    
    parameter_combinations = list(itertools.product(n_nodes, pca_n, epochs,
                                                    hidden_channels, order, k_neighbours))

    for p in parameter_combinations:

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
                        trial = savgol_filter(trial, 9,2)  
                    
                    # transform to time x channels
                    trial = trial.T
                    
                    # take all points except last
                    pos.append(trial)
                    
        # fit pca
        pca = PCA(n_components=p[1])
        pca.fit(np.vstack(pos))    
        
        # create empty list of lists for each condition
        pos = [[] for u in range(len(conditions))]
        vel = [[] for u in range(len(conditions))]      
          
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
                        trial = savgol_filter(trial, 9,2)  
                    
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
        data = utils.construct_dataset(pos, features=vel, graph_type='cknn', k=p[5], stop_crit=0.01, 
                                       n_nodes=p[0], n_workers=1, n_geodesic_nb=10, compute_cl=True, vector=False)
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
        
        par = {'epochs': p[2], #optimisation epochs
               'order': p[4], #order of derivatives
               'hidden_channels': p[3], #number of internal dimensions in MLP
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
            plt.savefig('./outputs/mds_embedding_nodes{}_pca{}_epochs{}_hc{}_o{}_k{}.png'.format(p[0],p[1],p[2],p[3],p[4],p[5]))

            plt.figure()
            plt.imshow(data.dist); plt.colorbar()  
            plt.savefig('./outputs/distance_matrix_nodes{}_pca{}_epochs{}_hc{}_o{}_k{}.png'.format(p[0],p[1],p[2],p[3],p[4],p[5]))
    
           # plt.figure()
           # plotting.embedding(data, data.y.numpy())
           # plt.savefig('./outputs/node_embedding_nodes{}_pca{}_epochs{}_hc{}_o{}_k{}.png'.format(p[0],p[1],p[2],p[3],p[4],p[5]))
    
        if save:
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


