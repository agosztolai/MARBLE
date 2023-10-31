import numpy as np
import matplotlib.pyplot as plt

conditions=['DownLeft','Left','UpLeft','Up','UpRight','Right','DownRight']  

def train_OLE(data, trial_ids, representation='lfads_factors'):
    
    X, Z = [], []
    unique_trial_ids = np.unique(trial_ids)
    for tr in unique_trial_ids:
        X.append(data[tr]['kinematics'])
        Z.append(data[tr][representation])

    X, Z = np.hstack(X), np.hstack(Z)

    X = X[:4,:] # take first four rows of kinematics (pos_x, pos_y, vel_x, vel_y)
    Z = np.vstack([Z,np.ones(Z.shape[1])])
    
    out  = np.linalg.lstsq(Z.T, X.T, rcond=None)
    Lw = out[0].T
    
    return Lw
    
    
def decode_kinematics(data, L, dt=20, representation='lfads_factors'):
    
    trial_emb = data[representation] # get trial embedding
    trial_kinematics = data['kinematics'] # get kinematics associated with trial

    trial_emb = np.vstack([trial_emb, np.ones(trial_emb.shape[1])])

    # empty array for decoding predictions
    trial_pred = np.empty([4, trial_kinematics.shape[1]])
    trial_pred[:] = np.nan

    trial_pred[:2,0] = trial_kinematics[:2,0] #first two entries are the x,y coordinates

    # predict velocity 
    z = np.matmul(L, trial_emb[:,0]); # decode
    trial_pred[[2,3],0] = z[[2,3]] #velocities

    # loop over each time point in trial
    for nt in range(1,trial_kinematics.shape[1]):

        neural = trial_emb[:,nt] # next point of embedding
        z = np.matmul(L, neural) # decode

        #trial_pred[:2,nt] = (1-alpha) * z[:2] + alpha * (trial_pred[:2,nt-1] + z[[2,3]]*dt/1000)
        trial_pred[:2,nt] = trial_pred[:2,nt-1] + z[[2,3]]*dt/1000
        trial_pred[[2,3],nt] = z[[2,3]]

    return trial_pred

# define a function for computing R-squared of kinematics
def correlation(data, trial_ids, representation='lfads_factors'):
    
    X, Z = [], []
    for tr in trial_ids:
        X.append(data[tr]['kinematics'])
        Z.append(data[tr][representation][:,:])

    X, Z = np.hstack(X), np.hstack(Z)

    r2_vel = np.mean([calcR2(X[2,:], Z[2,:]), calcR2(X[3,:], Z[3,:])])
    r2_pos = np.mean([calcR2(X[0,:], Z[0,:]), calcR2(X[1,:], Z[1,:])])
   
    return r2_pos, r2_vel

def calcR2(data, model):
    
    datavar = sum((data-np.mean(data))**2);
    errorvar = sum((model-data)**2);
    r2 = 1-errorvar/datavar   
    
    return r2

def plot_kinematics(data, session, trial_ids, representation='lfads_factors', ax=None, sz = 140):
    colors = plt.cm.viridis(np.linspace(0,1,7))        
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    for c,cond in enumerate(conditions):   
        for t in trial_ids:
            if data[session][t]['condition']==cond:
                meh = data[session][t][representation]
                ax.plot(meh[0,:],meh[1,:],c=colors[c])
                
    ax.set_title(representation)
    ax.set_xlim([-sz, sz])
    ax.set_ylim([-sz, sz])
    ax.set_axis_off()
    
    return ax