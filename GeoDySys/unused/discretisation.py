#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.special as scp
import numpy.ma as ma
from sklearn.cluster import MiniBatchKMeans


def kmeans_part(X,k,batchsize=None):
    
    if batchsize==None:
        batchsize = k*5
    if ma.count_masked(X)>0:
        labels = ma.zeros(X.shape[0],dtype=int)
        labels.mask = np.any(X.mask,axis=1)
        kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=k).fit(ma.compress_rows(X))
        labels[~np.any(X.mask,axis=1)] = kmeans.labels_
    else:
        kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=k).fit(X)
        labels=kmeans.labels_
        
    sizes = []
    for i in range(k):
        s = X[labels==i]
        sizes.append(s.max(0) - s.min(0))

    return kmeans.cluster_centers_, np.array(sizes), labels


def maxent_part(X, D, info_thresh):
    '''
    Cluster data by maximising the information in each cluster
    (Based on T. L. Carroll and J. M. Byers, PRE, 2016)
    
    INPUT
    -----
    data : 
        data vector
    D : 
        dimension of the data
        
    OUTPUT
    ------
    Xmean : numpy array
        mean coordinate of each cluster
    Xstd : numpy array
        sd of each cluster
    centers : list or integers
        index of point (in data) whose neighbours form clusters
    dlist : list or arrays
        list of distances from center to points in neighbourhood
    '''
    
    n = X.shape[0] # length of embedded signal
    unused_pts = [i for i in range(n)] # list of points used so far for clustering
    sizes = np.zeros([n,D])
    centers = []
    labels = np.zeros(n)
    sizes = []
    
    l=0
    while n-len(unused_pts) < 0.95*n: #loop until 95% of points used
                
        # pick next center from unused points
        center = np.random.choice(unused_pts)
                
        # find distance from center point to all unused points points
        Xu = X[unused_pts].copy()
        dist = pairwise_distances(Xu, X[[center]]).flatten()
        dist_list = np.argsort(dist,axis = 0)
                
        # Theiler exclusion: exclude points that are consecutive in time       
        diffs = np.diff(dist_list, axis = 0)
        dlt = np.where(np.abs(diffs) == 1)[0]
        dist_list = [i for i in dist_list if (i not in dlt)]#np.delete(dist_list, dlt) 
        
        k = nhoodsize(Xu,dist_list,D,info_thresh)
        
        dist_list = [unused_pts[i] for i in dist_list]
        unused_pts = [i for i in unused_pts if (i not in dist_list[:k])]
          
        centers.append(X[center])
        labels[dist_list[:k]] = l
        Xk = X[dist_list[:k]]
        sizes.append(Xk.max(0) - Xk.min(0))
        l=+1
        
    return centers, sizes, labels


def nhoodsize(embed_data,dist_list,D,info_thresh):
    '''
    Find number of nearest neighbours to a point in a given dimension D such 
    that the information in the bin is larger than what would be expected if
    the data was uniformly distributed.
    
    INPUT
    -----
    embed_data :  numpy array
        delay-embedded data
        
    dist_list : 1D numpy array
        list of pairwise distance from center to all other points
        
    D : int
        dimension of dataset
        
    info_thresh : float
        threshold of information comaprion
            
    OUTPUT
    ------
    k : int
        number of nearest neighbours to include in cluster
    '''
    n_parti = 2**D
    k = D + 1 # minimum # points in cluster - then increase k 
                # i.e., nhood size until info condition is violated
    dkfunc = 0 # information function - partition cost
    n = embed_data.shape[0] # length of embedded signal
    while dkfunc < info_thresh and k < n-1:
        k = k + 1
                    
        dk = dist_list[:k]
        #diagonal points
        xmax = np.max(embed_data[dk,:],0) #vertex with lowest coords
        xmin = np.min(embed_data[dk,:],0) #vertex with highest coords
                            
        if not (xmax==xmin).all(): # check to make sure max and min are not equal
            np_vec = np.zeros([k,1])
                        
            # turn point locations into box numbers
            for kd in range(D):
                np_vec += np.floor((embed_data[dk,kd:kd+1] - xmin[kd]) / (xmax[kd] - xmin[kd]))*2**kd
                      
            # np_vec += 1

            prob_vec = np.zeros([n_parti,1]) #data distribution over partitions
            for kp in range(n_parti):
                prob_vec[kp] = (np_vec == kp).sum() 
                      
            rho0 = k/n_parti #uniform distribution over partitions
            dKL = 0 
            for kp in range(n_parti):
                dKL = dKL + (prob_vec[kp] - rho0)*scp.digamma(prob_vec[kp] + 0.5) \
                          - scp.gammaln(prob_vec[kp] + 0.5) \
                          + scp.gammaln(rho0 + 0.5)
                      
            dKL = dKL/np.sqrt(2)
            dkfunc = dKL - n_parti*np.log2(n_parti) #penalty
            
    return k
