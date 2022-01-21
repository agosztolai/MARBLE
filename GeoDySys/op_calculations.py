import numpy as np
# import numpy.ma as ma
# from scipy.sparse import csc_matrix as sparse_matrix
# from scipy.sparse.linalg import eigs
# from scipy.linalg import eig
from scipy.sparse import diags,identity,coo_matrix
import msmtools.estimation as msm_estimation
# import msmtools.analysis as msm_analysis
# import stats
# import matplotlib.pyplot as plt
from GeoDySys.time_series import valid_flows, generate_flow


def get_transition_matrix(t_sample,labels,T,return_connected=False):
    count_matrix = get_count_matrix(t_sample,labels,T)
    connected_count_matrix = msm_estimation.connected_cmatrix(count_matrix)
    P = msm_estimation.tmatrix(connected_count_matrix)
    if return_connected:
        lcs = msm_estimation.largest_connected_set(count_matrix)
        return lcs,P
    else:
        return P


def get_count_matrix(t_ind,labels,T=1):
    # observable_seqs = ma.compress_rows(ma.vstack([labels[:-T],labels[T:]]).T)
    
    ts = np.arange(0,len(labels)-T)
    tt = np.arange(T,len(labels))
    ts, tt = valid_flows(t_ind, ts, tt)
    
    #eliminate trajectories that intersect target set multiple times
    flows, ts, tt = generate_flow(np.arange(len(t_ind)), ts, tt)
    flows = np.array(flows)
    notfirst = ((flows-flows[:,[-1]])==0).sum(1)>1
    ts.mask = ts.mask*notfirst
    tt.mask = tt.mask*notfirst

    row = labels[ts[~ts.mask]]#observable_seqs[:,0]
    col = labels[tt[~tt.mask]]#observable_seqs[:,1]

    data = np.ones(ts.count())
    C = coo_matrix((data, (row, col)), shape=(max(labels)+1, max(labels)+1))
    count_matrix = C.tocsr()
    
    return count_matrix

#### CODE BELOW IS FROM Costa, et al. 2021 arxiv 

# def segment_maskedArray(tseries,min_size=50):
#     '''
#     Segments  time series in case it has missing data
#     '''
#     if len(tseries.shape)>1:
#         mask = ~np.any(tseries.mask,axis=1)
#     else:
#         mask = ~tseries.mask
#     segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
    
#     return segments


# def get_count_ms(dtrajs,delay,nstates):
#     if len(dtrajs.shape)>1:
#         count_ms = coo_matrix((nstates,nstates))
#         for dtraj in dtrajs:
#             try:
#                 count_ms+=get_count_matrix(dtraj,delay,nstates)
#             except:
#                 print('Warning! No samples.')
#                 continue
#     else:
#         try:
#             count_ms=get_count_matrix(dtrajs,delay,nstates)
#         except:
#             print('Warning! No samples.')
            
#     return count_ms


# def tscales_samples(labels,delay,dt,size,n_modes=5,reversible=True):
#     dtrajs = get_split_trajs(labels,size)
#     nstates = np.max(labels)+1
#     P_traj=[]
#     ts_traj = []
#     for sample_traj in dtrajs:
#         count_ms = get_count_ms(sample_traj,delay,nstates)
#         connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
#         P = msm_estimation.tmatrix(connected_count_matrix)
#         if reversible:
#             R = get_reversible_transition_matrix(P)
#             tscale = compute_tscales(R,delay,dt,k=n_modes+1)
#         else:
#             tscale = compute_tscales(P,delay,dt,k=n_modes+1)
#         ts_traj.append(tscale)
#         P_traj.append(P)
        
#     return ts_traj,P_traj


# def get_connected_labels(labels,lcs):
#     final_labels = ma.zeros(labels.shape,dtype=int)
#     for key in np.argsort(lcs):
#         final_labels[labels==lcs[key]]=key+1
#     final_labels[final_labels==0] = ma.masked
#     final_labels-=1
    
#     return final_labels

    
# def sorted_spectrum(R,k=5,which='LR'):
#     eigvals,eigvecs = eigs(R,k=k,which=which)
#     sorted_indices = np.argsort(eigvals.real)[::-1]
#     return eigvals[sorted_indices],eigvecs[:,sorted_indices]


# def compute_tscales(P,delay,dt=1,k=2):
#     try:
#         if P.shape[1]<=10:
#             eigvals = np.sort(eig(P.toarray())[0])[::-1][:k]
#         else:
#             eigvals = eigs(P,k=k,which='LR',return_eigenvectors=False)
#         sorted_indices = np.argsort(eigvals.real)[::-1]
#         eigvals = eigvals[sorted_indices][1:].real
#         eigvals[np.abs(eigvals-1)<1e-12] = np.nan
#         eigvals[eigvals<1e-12] = np.nan
#         return -(delay*dt)/np.log(np.abs(eigvals))
    
#     except:
#         return np.array([np.nan]*(k-1))
    

# def get_reversible_transition_matrix(P):
#     probs = stationary_distribution(P)
#     P_hat = diags(1/probs)*P.transpose()*diags(probs)
#     R=(P+P_hat)/2
#     return R
    
# def get_split_trajs(labels,size = 0):
#     if size == 0:
#         size = len(labels)/20
#     return ma.array([labels[kt:kt+size] for kt in range(0,len(labels)-size,size)])


# def implied_tscale(labels,size,delay,dt,n_modes,reversible=True):
#     dtrajs = get_split_trajs(labels,size)
#     nstates = np.max(labels)+1
#     count_ms = get_count_ms(dtrajs,delay,nstates)
#     connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
#     P = msm_estimation.tmatrix(connected_count_matrix)
#     if reversible:
#         R = get_reversible_transition_matrix(P)
#         tscale = compute_tscales(R,delay,dt,k=n_modes+1)
#     else:
#         tscale = compute_tscales(P,delay,dt,k=n_modes+1)
#     return tscale


# def get_bootstrapped_Ps(labels,delay,n_samples,size = 0):
#     #get dtrajs to deal with possible nans
#     dtrajs = get_split_trajs(labels,size)
#     nstates = np.unique(labels.compressed()).shape[0]
    
#     sample_Ps=[]
#     for k in range(n_samples):
#         sample_trajs = [dtrajs[k] for k in np.random.randint(0,len(dtrajs),len(dtrajs))]
#         count_ms = get_count_ms(sample_trajs,delay,nstates)
#         connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
#         P = msm_estimation.tmatrix(connected_count_matrix)
#         sample_Ps.append(P)
#     return sample_Ps
        
    
# def boostrap_tscales(labels,delay,dt,n_modes,n_samples = 1000,size=0,reversible=True):
#     Ps = get_bootstrapped_Ps(labels,delay,n_samples,size)
#     tscales=np.zeros((n_samples,n_modes))
#     for k,P in enumerate(Ps):
#         if reversible:
#             R = get_reversible_transition_matrix(P)
#             tscale = compute_tscales(R,delay,dt,k=n_modes+1)
#         else:
#             tscale = compute_tscales(P,delay,dt,k=n_modes+1)
#         tscales[k,:]=tscale
#     return tscales
    
    
# def bootstrap_tscale_sample(labels,delay,dt,n_modes,size=0,reversible=True):
#     dtrajs = get_split_trajs(labels,size)
#     nstates = np.unique(labels.compressed()).shape[0] 
    
#     sample_trajs = [dtrajs[k] for k in np.random.randint(0,len(dtrajs),len(dtrajs))]
#     count_ms = get_count_ms(sample_trajs,delay,nstates)
#     connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
#     P = msm_estimation.tmatrix(connected_count_matrix)
#     if reversible:
#         R = get_reversible_transition_matrix(P)
#         tscale = compute_tscales(R,delay,dt,k=n_modes+1)
#     else:
#         tscale = compute_tscales(P,delay,dt,k=n_modes+1)
#     return tscale


# def bootstrap_tscales_delays(range_delays,labels,n_modes,dt,n_samples=1000,size=0,reversible=True):
#     dtrajs = get_split_trajs(labels,size)
#     nstates = np.unique(labels.compressed()).shape[0]    
#     sample_trajs = [dtrajs[k] for k in np.random.randint(0,len(dtrajs),len(dtrajs))]
#     tscales=np.zeros((len(range_delays),n_modes))
#     for kd,delay in enumerate(range_delays):
#         try:
#             count_ms = get_count_ms(sample_trajs,delay,nstates)
#             connected_count_matrix = msm_estimation.connected_cmatrix(count_ms)
#             P = msm_estimation.tmatrix(connected_count_matrix)
#             if reversible:
#                 R = get_reversible_transition_matrix(P)
#                 tscale = compute_tscales(R,delay,dt,k=n_modes+1)
#             else:
#                 tscale = compute_tscales(P,delay,dt,k=n_modes+1)
#             tscales[kd,:] = tscale
#         except:
#             continue
#     return tscales


# def compute_implied_tscales(labels,range_delays,dt=1,n_modes=5,n_samples=1000,size=0,reversible=False,confidence = 95):
#     if size==0:
#         size = np.max(range_delays)*2
#     cil = (100-confidence)/2
#     ciu = 100-cil
#     cil_delay=np.zeros((len(range_delays),n_modes))
#     ciu_delay=np.zeros((len(range_delays),n_modes))
#     mean_delay=np.zeros((len(range_delays),n_modes))
#     bootstrapped_tscales  = []
#     for kd,delay in enumerate(range_delays):
#         mean_tscale = implied_tscale(labels,size,delay,dt,n_modes,reversible)
#         tscales_samples = boostrap_tscales(labels,delay,dt,n_modes,n_samples,size,reversible)
#         mean_delay[kd,:] = mean_tscale
#         cil_delay[kd,:] =  np.nanpercentile(tscales_samples,cil,axis=0)
#         ciu_delay[kd,:] = np.nanpercentile(tscales_samples,ciu,axis=0)
#     return cil_delay,ciu_delay,mean_delay


# def find_asymptote(ts,tol=1e-5):
#     es_0 = np.percentile(ts,80)
#     tau_0 = np.where(np.diff(np.sign(ts-es_0)))[0][0]
#     es_list=[es_0,]
#     tau_list=[tau_0,]
#     m,b=np.polyfit(np.arange(tau_0,len(ts)),np.cumsum(ts)[tau_list[-1]:],1)
#     es_list.append(m)
#     tau_list.append(np.where(np.diff(np.sign(ts-m)))[0][0])
#     while es_list[-1]-es_list[-2]>tol:
#         m,b=np.polyfit(np.arange(tau_list[-1],len(ts)),np.cumsum(ts)[tau_list[-1]:],1)
#         es_list.append(m)
#         try:
#             tau_list.append(np.where(np.diff(np.sign(ts-m)))[0][0])
#         except:
#             break
#     return tau_list[-1],es_list[-1]


# def stationary_distribution(P):
#     probs = msm_analysis.statdist(P)
#     return probs


# def get_entropy(labels):
#     #get dtrajs to deal with possible nans
#     P = get_transition_matrix(labels,1)
#     probs = stationary_distribution(P)
#     logP = P.copy()
#     logP.data = np.log(logP.data)
#     return (-diags(probs).dot(P.multiply(logP))).sum()



# def simulate(P,state0,iters):
#     '''
#     Monte Carlo simulation of the markov chain characterized by the matrix P
#     state0: initial system
#     iters: number of iterations of the simulation
#     '''
#     states = np.zeros(iters,dtype=int)
#     states[0]=state0
#     state=state0
#     for k in range(1,iters):
#         new_state = np.random.choice(np.arange(P.shape[1]),p=list(np.hstack(P[state,:].toarray())))
#         state=new_state
#         states[k]=state
#     return states


# def state_lifetime(states,tau):
#     '''
#     Get distribution of lifetimes of each state in states
#     tau is the sampling time of the states
#     '''
#     durations=[]
#     for state in np.sort(np.unique(states.compressed())):
#         gaps = states==state
#         gaps_boundaries = np.where(np.abs(np.diff(np.concatenate([[False], gaps, [False]]))))[0].reshape(-1, 2)
#         durations.append(np.hstack(np.diff(gaps_boundaries))*tau)
#     return durations


# from scipy.signal import find_peaks


# def optimal_partition(phi2,inv_measure,P,return_rho = True):
    
#     X = phi2
#     c_range = np.sort(phi2)[1:-1]
#     rho_c = np.zeros(len(c_range))
#     rho_sets = np.zeros((len(c_range),2))
#     for kc,c in enumerate(c_range):
#         labels = np.zeros(len(X),dtype=int)
#         labels[X<=c] = 1
#         rho_sets[kc] = [(inv_measure[labels==idx]*(P[labels==idx,:][:,labels==idx])).sum()/inv_measure[labels==idx].sum() 
#                       for idx in range(2)]
#     rho_c = np.min(rho_sets,axis=1)
#     peaks, heights = find_peaks(rho_c, height=0.5) 
#     if len(peaks)==0: #lower height
#         print('No prominent coherent set')
#         return None
#     else:
#         idx = peaks[np.argmax(heights['peak_heights'])]
        
#         c_opt = c_range[idx]
#         kmeans_labels = np.zeros(len(X),dtype=int)
#         kmeans_labels[X<c_opt] = 1

#         if return_rho:
#             return c_range,rho_sets,idx,kmeans_labels
#         else:
#             return kmeans_labels


# def subdivide_state_optimal(phi2,kmeans_labels,inv_measure,P,indices,plot):
#     if plot:
#         c_range,rho_sets,idx,labels_ =  optimal_partition(phi2,inv_measure,P,return_rho=True)
#         kmeans_labels[indices] = labels_+np.max(kmeans_labels)+1
#         plt.figure(figsize=(5,5))
#         plt.scatter(c_range,rho_sets[:,0],s=10)
#         plt.scatter(c_range,rho_sets[:,1],s=10)
#         rho_c = np.min(rho_sets,axis=1)
#         plt.plot(c_range,rho_c,c='k',ls='--')
#         plt.scatter(c_range[idx],rho_c[idx],c='r',marker='x')
#         plt.ylim(.2,1)
#         plt.xlabel(r'$\phi_2$',fontsize=15)
#         plt.ylabel(r'$\rho$',fontsize=15)
#         plt.xticks(fontsize=12)
#         print(len(np.unique(kmeans_labels)))
#         plt.show()
        
#     else:
#         kmeans_labels[indices] = optimal_partition(phi2,inv_measure,P,return_rho=False)+np.max(kmeans_labels)+1

#     final_kmeans_labels = np.zeros(kmeans_labels.shape,dtype=int)
#     for new_idx,label in enumerate(np.sort(np.unique(kmeans_labels))):
#         final_kmeans_labels[kmeans_labels==label]=new_idx

#     return final_kmeans_labels


# def recursive_partitioning_optimal(final_labels,delay,phi2,inv_measure,P,n_final_states,plot=False,save=False):
#     c_range,rho_sets,idx,kmeans_labels =  optimal_partition(phi2,inv_measure,P,return_rho=True)

#     if plot:
#         plt.figure(figsize=(5,5))
#         plt.scatter(c_range,rho_sets[:,0],s=10)
#         plt.scatter(c_range,rho_sets[:,1],s=10)

#         rho_c = np.min(rho_sets,axis=1)
#         plt.plot(c_range,rho_c,c='k',ls='--')
#         plt.scatter(c_range[idx],rho_c[idx],c='r',marker='x')
#         plt.ylim(.3,1)
#         # plt.xlim(-0.04,0.04)
#         plt.xlabel(r'$\phi_2$',fontsize=15)
#         plt.ylabel(r'$\rho$',fontsize=15)
#         plt.xticks(fontsize=12)
#         print(len(np.unique(kmeans_labels)))
#         if save:
#             plt.savefig('rho_{}states_Foraging.pdf'.format(len(np.unique(kmeans_labels))))
#         plt.show()
    
#     labels_tree=np.zeros((n_final_states,len(kmeans_labels)),dtype=int)
#     labels_tree[0,:] = kmeans_labels
#     k=1
#     for k in range(1,n_final_states):
#         print(k)
#         lambda_2=[]
#         eigfunctions_states=[]
#         indices_states=[]
#         im_states=[]
#         P_states=[]
#         for state in np.unique(kmeans_labels):
#             cluster_traj = ma.zeros(final_labels.shape,dtype=int)
#             cluster_traj[~final_labels.mask] = np.array(kmeans_labels)[final_labels[~final_labels.mask]]
#             cluster_traj[final_labels.mask] = ma.masked
#             labels_here = ma.zeros(final_labels.shape,dtype=int)
#             sel = cluster_traj==state
#             labels_here[sel] = final_labels[sel]
#             labels_here[~sel] = ma.masked

#             lcs,P = get_transition_matrix(labels_here,delay,return_connected=True)
#             R = get_reversible_transition_matrix(P)
#             im = stationary_distribution(P)
#             eigvals,eigvecs = sorted_spectrum(R,k=2)
#             indices = np.zeros(len(np.unique(final_labels.compressed())),dtype=bool)
#             indices[lcs] = True

#             eigfunctions_states.append((eigvecs.real/np.linalg.norm(eigvecs.real,axis=0))[:,1])
#             indices_states.append(indices)
#             lambda_2.append(eigvals[1].real)
#             P_states.append(P)
#             im_states.append(im)


#         measures = [(inv_measure[kmeans_labels==state]).sum() for state in np.unique(kmeans_labels)]
#         state_to_split = np.argmax(measures)
#         print(state_to_split,lambda_2,measures)
#         kmeans_labels = subdivide_state_optimal(eigfunctions_states[state_to_split],
#                                                kmeans_labels,im_states[state_to_split],P_states[state_to_split],
#                                                indices_states[state_to_split],plot)
        
#         labels_tree[k,:] = np.copy(kmeans_labels)
#         k+=1

#     return labels_tree    
