#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def stack(X):
    """
    Stak ensemble of trajectories into attractor

    Parameters
    ----------
    X : list[np.array)]
        Individual trajectories in separate lists.

    Returns
    -------
    X_stacked : np.array
        Trajectories stacked.

    """
    
    X_stacked = np.vstack(X)
    
    return X_stacked


def unstack(X, t_sample):
    """
    Unstack attractor into ensemble of individual trajectories.

    Parameters
    ----------
    X : np.array
        Attractor.
    t_sample : list[list]
        Time indices of the individual trajectories.

    Returns
    -------
    X_unstack : list[np.array]
        Ensemble of trajectories.

    """
    
    X_unstack = []
    for t in t_sample:
        X_unstack.append(X[t,:])
        
    return X_unstack


def standardize_data(X, axis=0):
    """
    Normalize data

    Parameters
    ----------
    X : nxd array (dimensions are columns!)
        Coordinates of n points on a manifold in d-dimensional space..
    axis : 0,1, optional
        Dimension to normalize. The default is 0 (along dimensions).

    Returns
    -------
    X : nxd array (dimensions are columns!)
        Normalized data.

    """
    
    X -= np.mean(X, axis=axis, keepdims=True)
    X /= np.std(X, axis=axis, keepdims=True)
        
    return X
