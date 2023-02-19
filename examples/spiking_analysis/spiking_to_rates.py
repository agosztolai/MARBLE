import numpy as np

import sys

import matplotlib.pyplot as plt

import neo
from elephant.statistics import instantaneous_rate
from elephant.kernels import GaussianKernel, ExponentialKernel
from quantities import ms

import mat73

from tqdm import tqdm

import pickle

def main():
    """
    Converts matlab spiking data into instantaneous rates in a suitable format for further analysis
    """

    folder = '/media/robert/Extreme SSD/ResearchProjects/MARBLE/lfads-neural-stitching-reproduce/'
    file = 'conditions_spiking_data.mat'
    
    # load data compiled into matlab cell array
    data = mat73.loadmat(folder+file)['all_results']
    
    # defining a dictionary
    rates = {}
    
    # defining conditions by their ordering (this was how it was ordered in matlab script)
    conditions = ['DownLeft','Left','UpLeft','Up','UpRight','Right','DownRight']
    
    # loop over each daily session 
    for d, data_day in enumerate(tqdm(data)):
        
        # define empty dictionary for each day
        rates[d] = {}
        
        # loop over the 7 conditions
        for c, cond in enumerate(conditions):
            
            # define empty list for each condition during each session
            trial_data = []
            
            # extracting data for a single condition on a single day (composed of t trials)
            data_day_cond = data_day[c]
    
            # loop over trials
            for t, trial in enumerate(data_day_cond):
                
                # if the trial exists (sometimes there is None)
                if trial:
                    trial = trial[0] # it was a single element list
                    
                    # loop over neurons
                    inst_rates = []
                    for ch in range(trial.shape[0]):
                        
                        # extract spikes for a given channel (neuron)
                        spikes = np.where(trial[ch,:])[0]
                        
                        # get spike train (1200 milliseconds)
                        st = neo.SpikeTrain(spikes, units='ms', t_stop=1200)
                        
                        # get rates
                        gk = GaussianKernel(100*ms) # increase this for smoother signals (previously used auto)
                        # ek = ExponentialKernel(100*ms) # assymetric kernel and not smooth output
                        inst_rate = instantaneous_rate(st, kernel=gk, sampling_period=50*ms).magnitude
                        
                        # append into list
                        inst_rates.append(inst_rate)
                    
                    # stack rates back together and transpose = (channels by time)
                    inst_rates = np.hstack(inst_rates).T
                    
                    # append rates from one trial into trial data
                    trial_data.append(inst_rates)
                    
            # stack into an array of trial x channels x time
            trial_data = np.dstack(trial_data).transpose(2,0,1)
            
            # storing all trials by session and condition
            rates[d][cond] = trial_data
            
    with open('rate_data.pkl', 'wb') as handle:
        pickle.dump(rates, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    

if __name__ == '__main__':
    sys.exit(main())



