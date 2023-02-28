#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from MARBLE import net

data, days, conditions = pickle.load(open('../outputs/spiking_data/data_dataobject_k40.pkl','rb'))

par = {'epochs': 100, #optimisation epochs
       'order': 2, #order of derivatives
       'hidden_channels': 64, #number of internal dimensions in MLP
       'out_channels': 5,
       'inner_product_features': False,
       'vec_norm': False,
       'diffusion': False,
       }
    
model = net(data, **par)
    
model.run_training(data, use_best=True, outdir='../outputs/spiking_data/k40')  
