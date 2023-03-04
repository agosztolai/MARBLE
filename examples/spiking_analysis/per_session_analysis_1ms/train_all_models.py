#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import sys
from MARBLE import net, postprocessing


def main():    
    
    """ fitting model for each day + pca embedding """    
    
    # instantaneous rate data
    rates = pickle.load(open('../../outputs/spiking_data/rate_data_1ms.pkl','rb'))       
    
    # list of days
    days = rates.keys()
    
    
    # loop over each day
    for day in days:

        # load data for marble
        data = pickle.load(open('../../outputs/spiking_data/rate_data_separate_manifolds/data_object_session_{}.pkl'.format(day),'rb'))
        

        par = {'epochs': 150, #optimisation epochs
               'order': 2, #order of derivatives
               'hidden_channels': 32, #number of internal dimensions in MLP
               'out_channels': 8,
               'inner_product_features': False,
               'diffusion': True,
               }
        
        model = net(data, par=par)
        
        model.run_training(data, use_best=True, outdir='../../outputs/spiking_data/rate_data_separate_manifolds/session_{}'.format(day))        
        data = model.evaluate(data)   
        data = postprocessing(data, n_clusters=50)
        
        with open('../../outputs/spiking_data/rate_data_separate_manifolds/data_object_session_{}.pkl'.format(day,day), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_vector_array(coords):
    """ function for defining the vector features from each array of coordinates """
    diff = np.diff(coords, axis=0)
    return diff

if __name__ == '__main__':
    sys.exit(main())


