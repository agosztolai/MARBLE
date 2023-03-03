import numpy as np
import sys
import neo
from elephant.statistics import instantaneous_rate
from elephant.kernels import GaussianKernel
from quantities import ms
from MARBLE import utils
import mat73
import pickle

def main():
    
    folder = '../outputs/spiking_data/'#'/media/robert/Extreme SSD/ResearchProjects/MARBLE/lfads-neural-stitching-reproduce/'
    file = 'conditions_spiking_data.mat'
    
    # load data compiled into matlab cell array
    data = mat73.loadmat(folder+file)['all_results']
    
    rates = utils.parallel_proc(spikes_to_rates, 
                                range(len(data)), 
                                data,
                                processes=-1,
                                desc="Converting spikes to rates...")
        
    all_rates = {}
    for i, rates_day in enumerate(rates):
        all_rates[i] = rates_day
        
    
    with open('../outputs/spiking_data/rate_data.pkl', 'wb') as handle:
        pickle.dump(all_rates, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def spikes_to_rates(data, d):
    """
    Converts matlab spiking data into instantaneous rates in a suitable format for further analysis
    """
    
    # defining conditions by their ordering (this was how it was ordered in matlab script)
    conditions = ['DownLeft','Left','UpLeft','Up','UpRight','Right','DownRight']
    
    data_day = data[d] #daily session 
        
    gk = GaussianKernel(100*ms) # increase this for smoother signals (previously used auto)
    
    # define empty dictionary for each day
    rates = {}
        
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
                    # ek = ExponentialKernel(100*ms) # assymetric kernel and not smooth output
                    inst_rate = instantaneous_rate(st, kernel=gk, sampling_period=1*ms).magnitude
                        
                    # append into list
                    inst_rates.append(inst_rate.flatten())
                    
                # stack rates back together and transpose = (channels by time)
                inst_rates = np.stack(inst_rates,axis=1)
                    
                # append rates from one trial into trial data
                trial_data.append(inst_rates)
                    
        # stack into an array of trial x channels x time
        rates[cond] = np.dstack(trial_data).transpose(2,0,1)
        
    return rates
           
            
if __name__ == '__main__':
    sys.exit(main())