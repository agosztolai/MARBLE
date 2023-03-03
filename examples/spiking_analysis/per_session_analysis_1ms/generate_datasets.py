
import numpy as np

import pickle
import sys

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, KernelPCA

from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, MDS

from sklearn.neighbors import LocalOutlierFactor

from MARBLE import utils

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
    pca_n = 3
    
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
                
                # apply transformation to single trial
                trial = pca.transform(trial)
                
                # take all points except last
                pos[c].append(trial[:-1,:])
                
                # extract vectors between coordinates
                vel[c].append(get_vector_array(trial))
                
                timepoints[c].append(np.linspace(0,trial.shape[0]-2,trial.shape[0]-1))
                condition_labels[c].append(np.repeat(c,trial.shape[0]-1))
           
        # stack the trials within each condition
        pos = [np.vstack(u) for u in pos] # trials x time x channels
        vel = [np.vstack(u) for u in vel] # trials x time x channels
        timepoints = [np.hstack(u) for u in timepoints] 
        condition_labels = [np.hstack(u) for u in condition_labels]   
            
        # construct data for marble
        # data = utils.construct_dataset(pos, features=vel, graph_type='cknn', k=30, stop_crit=0.02,
        #                                n_geodesic_nb=10, compute_laplacian=True, vector=False)
        
        # extracting the time points associated with data
        # for c,cond in enumerate(conditions):
        #     time = timepoints[c]
        #     time = time[data.sample_ind[data.y==c]]
        #     timepoints[c] = time
        
        # data.time = timepoints
        
        with open('../../outputs/spiking_data/raw_data_session_{}_3D.pkl'.format(day), 'wb') as handle:
            pickle.dump([pos, vel, timepoints, condition_labels], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # with open('../../outputs/spiking_data/data_object_session_{}.pkl'.format(day), 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_vector_array(coords):
    """ function for defining the vector features from each array of coordinates """
    diff = np.diff(coords, axis=0)
    return diff

if __name__ == '__main__':
    sys.exit(main())


