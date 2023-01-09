from scipy.io import loadmat
import numpy as np
import sys
import neo
from elephant.statistics import instantaneous_rate
from quantities import ms
import pickle


def main():
        
    file = '../data/conditions_spiking_data.mat'
    
    #data is a matrix with shape (trials, conditions)
    data = loadmat(file)['result']
    
    conditions=['DownLeft','DownRight','Left','Right','UpLeft','UpRight','Up']
    
    rates = {}
    for i, cond in enumerate(conditions):
                
        rates_trial = []
        for t, trial in enumerate(data[:,i]):
            if trial.shape[1] != 0:                
                try:
                    rates_channel = []
                    #trial[0][0] is a matrix with shape (channel, timesteps)
                    for c, channel in enumerate(trial[0][0]):
                        spikes = np.where(channel)[0]
                        
                        st = neo.SpikeTrain(spikes, units='ms', t_stop=1200)
                        r = instantaneous_rate(st, sampling_period=50*ms).magnitude

                        rates_channel.append(r.flatten())
                                                                    
                    rates_trial.append(np.vstack(rates_channel))
                except:
                    continue
            
        #rates[cond] is a matrix with shape (trials, channels, timesteps)
        rates[cond] = np.stack(rates_trial, axis=0)
        
    pickle.dump(rates, open('../data/rate_data.pkl','wb'))

if __name__ == '__main__':
    sys.exit(main())



