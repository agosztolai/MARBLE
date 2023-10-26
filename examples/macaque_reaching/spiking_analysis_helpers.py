import numpy as np

def train_OLE(data, trial_ids, representation='lfads_factors'):
    
    X, Z = [], []
    unique_trial_ids = np.unique(trial_ids)
    for tr in unique_trial_ids:
        #print(data[tr]['kinematics'][:,:-1].shape, data[tr][representation].shape)
        X.append(data[tr]['kinematics'][:,:-1])
        Z.append(data[tr][representation])

    X, Z = np.hstack(X), np.hstack(Z)

    X = X[:4,:] # they only took first four rows of kinematics
    Z = np.vstack([Z,np.ones(Z.shape[1])])
    
    out  = np.linalg.lstsq(Z.T, X.T, rcond=None)
    Lw = out[0].T
    
    return Lw
    
def decode_kinematics(data, L, alpha=1, dt=20, representation='lfads_factors'):
        
    trial_emb = data[representation] # get trial embedding
    trial_kinematics = data['kinematics'][:,:-1] # get kinematics associated with trial

    trial_emb = np.vstack([trial_emb, np.ones(trial_emb.shape[1])])

    # first time point of embedding
    neural = trial_emb[:,0]

    # empty array for decoding predictions
    trial_pred = np.empty([trial_kinematics.shape[0]+1,trial_kinematics.shape[1]])
    trial_pred[:] = np.nan

    trial_pred[:2,0] = trial_kinematics[:2,0]

    # predict velocity 
    z  = np.matmul(L,neural); # decode
    trial_pred[[2,3],0] = z[[2,3]]
    trial_pred[4,0] = 1

    # loop over each time point in trial
    for nt in range(1,trial_kinematics.shape[1]):

        neural = trial_emb[:,nt] # next point of embedding
        z  = np.matmul(L,neural); # decode

        trial_pred[:2,nt] = (1-alpha)*z[:2] + alpha *(trial_pred[:2,nt-1] + z[[2,3]]*dt/1000);
        trial_pred[[2,3],nt] = z[[2,3]]
        trial_pred[4,nt] = 1

    return trial_pred

# define a function for computing R-squared of kinematics
def correlation(data, trial_ids, representation='lfads_factors'):
    
    X, Z = [], []
    for tr in trial_ids:
        X.append(data[tr]['kinematics'][:,:-1])
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