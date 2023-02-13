#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.append("./RNN_scripts")
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from RNN_scripts import dms, ranktwo, clustering
from RNN_scripts.modules import LowRankRNN, train
import seaborn as sns

import pickle

import MARBLE
from MARBLE import utils, geometry, plotting, postprocessing, compare_attractors
from example_utils import generate_trajectories, plot_experiment, aggregate_data


# In[2]:


matplotlib.rcParams['figure.figsize'] = (6, 5)
matplotlib.rcParams['font.size'] = 30
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.titlesize'] = 'medium'
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False


# # Load trained model

# In[3]:


noise_std = 5e-2
alpha = 0.2
hidden_size=500

x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
net =  LowRankRNN(2, hidden_size, 1, noise_std, alpha, rank=2)
net.load_state_dict(torch.load('./RNN_scripts/dms_rank2_500.pt', map_location='cpu'))
net.svd_reparametrization()

net2 =  LowRankRNN(2, hidden_size, 1, noise_std, alpha, rank=2)
net2.load_state_dict(torch.load('./RNN_scripts/dms_rank2_500_2.pt', map_location='cpu'))
net2.svd_reparametrization()



n_gains=20
stim1_begin, stim1_end, stim2_begin, stim2_end, decision = 25, 50, 200, 225, 275
epochs = [0, stim1_begin, stim1_end, stim2_begin, stim2_end, decision]
gain = np.linspace(1,0,n_gains)
    
input = torch.zeros(n_gains, decision, 2)
for i, g in enumerate(gain):
    input[i, stim1_begin:stim1_end, 0] = g
    input[i, stim2_begin:stim2_end, 0] = g
    
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
plt.plot(input[0])
#plt.savefig('./outputs/stim.svg')


# # Plot phase portraits of two different dynamics  

# In[5]:


n_traj=50

traj = generate_trajectories(net, input, epochs, n_traj, fname='./outputs/RNN_trajectories.pkl')
traj2 = generate_trajectories(net2, input, epochs, n_traj, fname='./outputs/RNN_trajectories_2.pkl')




pos, vel = aggregate_data(net, traj, epochs)


# In[ ]:


data = utils.construct_dataset(pos, features=vel, graph_type='cknn', k=15, stop_crit=0.02)


# In[ ]:


# titles = [r'$gain$ = {:0.2f}, {}'.format(g, s) for s in ['stim', 'unstim'] for g in gain ]

# plotting.fields(data, col=4, alpha=0.2, width=0.01, scale=200, titles=titles)
# plt.show()


# # # Create new network by fitting Gaussian mixture to the connectivity space

# # In[ ]:


# net_sampled = clustering.to_support_net(net, z, scaling=True)


# # Check that the resampled networks still give the same validation losses

# # In[ ]:


# accs2 = []
# for _ in range(10):
#     net_sampled.resample_basis()
#     loss, acc = dms.test_dms(net_sampled, x_val, y_val, mask_val)
#     accs2.append(acc)


# # Train net network for a few epochs

# # In[ ]:


# train(net_sampled, x_train, y_train, mask_train, 20, lr=1e-6, resample=True, keep_best=True, clip_gradient=1)


# # # Plot phase portraits of the resampled system for various parameters

# # In[ ]:


# n_traj=50

# traj_sampled = generate_trajectories(net_sampled, input, epochs, n_traj, fname='./outputs/RNN_trajectories_sampled.pkl')
# plot_experiment(net_sampled, input, traj_sampled, epochs)


# # In[ ]:


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

# par = {'epochs': 100, #optimisation epochs
#        'order': 2, #order of derivatives
#        'hidden_channels': 32, #number of internal dimensions in MLP
#        'out_channels': 3,
#        'inner_product_features': True,
#       }

# model = MARBLE.net(data, **par)
# model.run_training(data)


# # In[ ]:


# data = model.evaluate(data)
# n_clusters=20
# data = postprocessing(data, n_clusters=n_clusters)

# emb_MDS, _ = geometry.embed(data.dist, embed_typ = 'MDS')
# labels = [g for i in range(2) for g in gain ]
# plotting.embedding(emb_MDS, labels, s=30, alpha=1)


# # In[ ]:


# plt.imshow(data.dist)


# # In[ ]:


# titles = [r'$gain$ = {:0.2f}, {}'.format(g, s) for s in ['stim', 'unstim'] for g in gain ]
# labels = np.hstack([gain,gain])
# plotting.embedding(data.emb_2d, data.y.numpy(), titles=titles)


# # In[ ]:





# In[ ]:




