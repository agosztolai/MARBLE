#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from MARBLE import plotting, geometry, utils, net
from DE_library import simulate_ODE
import pyEDM as pyEDM
import matplotlib.pyplot as plt
import example_utils

import sys

def main():
    # # Simulate Lorenz system
    
    # In[5]:
    
    
    # par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}
    # fun = 'lorenz'
    
    # n=1000
    # T=50
    
    # #simulate system
    # x0 = [-8.0, 7.0, 27.0]
    # T = np.linspace(0, T, n)
    # mu, sigma = 0, 1 # mean and standard deviation
    # X, V = simulate_ODE(fun, T, x0, par, noise=False, mu=mu, sigma=sigma)
    
    # from sklearn import random_projection
    # transformer = random_projection.SparseRandomProjection(n_components=1)
    
    # n_obs = 5
    
    # x = []
    # for i in range(n_obs):
    #     x_tmp = transformer.fit_transform(X)
    #     x_tmp = utils.standardise(x_tmp, norm='max')
    #     x.append(x_tmp)
        
    # x_stacked = np.hstack(x)
    # df = utils.to_pandas(x_stacked)
    
    import pickle
    
    df = pickle.load(open('data.pkl','rb'))
    
    
    E = 3
    
    #find best embedding views
    columns = list(df.columns)[1:]
    MVE = pyEDM.Multiview(dataFrame = df, 
                          lib = [1, 100], 
                          pred = [100, 200], 
                          E = E, 
                          D = E,
                          columns = columns, 
                          target = columns[0], 
                          showPlot = True)
    
    views = MVE['View'].iloc[:,:E]
    views = np.array(views, dtype=np.int)
    
    #create embeddings
    embedding = pyEDM.Embed(dataFrame = df, E = E, tau = -1, columns = columns)
    embedding = np.array(embedding.dropna())
    all_embeddings = np.stack([embedding[:,v-1] for v in views], axis=2)
    
    
    #first create data with the best embedding to fit the graph
    best_embedding = all_embeddings[...,0]
    feature = np.diff(best_embedding, axis=0) #velocity vectors
    pos = best_embedding[:-1]
    data = utils.construct_dataset(pos, feature, graph_type='cknn', k=10)
    
    #now add all other embeddings
    n_emb = all_embeddings.shape[2]
    feature = [np.diff(all_embeddings[...,i], axis=0) for i in range(n_emb)]
    
    #bring it to form [feature_1(x_1),...,feature_n(x_1),  feature_1(x_2),...,feature_n(x_2)]
    feature = np.stack(feature, axis=2)
    feature = feature.reshape(feature.shape[0], -1)
    
    data.x = utils.np2torch(feature)
    data.num_node_features = feature.shape[1]
    
    
    
    par = {'batch_size': 256, #batch size, this should be as large as possible
            'epochs': 30, #optimisation epochs
            'order': 2, #order of derivatives
            'n_lin_layers': 2,
            'hidden_channels': 16, #number of internal dimensions in MLP
            'out_channels': 4,
            'diffusion': False,
            'inner_product_features': True,
            'vector' : True
          }
    
    model = net(data, **par)
    #model.run_training(data)


if __name__ == '__main__':
    sys.exit(main())